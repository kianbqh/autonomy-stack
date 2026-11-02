#include "slam/core.hpp"
#include <Eigen/Eigenvalues>
#include <Eigen/Cholesky>
#include <nanoflann.hpp>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <limits>
#include <tuple>
#include <stdexcept>
#if NANOFLANN_VERSION != 0x163
#error "Track C requires nanoflann 1.6.3"
#endif

namespace slam {
namespace {
struct Adapter {
  const Cloud& p;
  size_t kdtree_get_point_count() const {return p.size();}
  double kdtree_get_pt(size_t i,size_t d) const {return p[i][d];}
  template<class B> bool kdtree_get_bbox(B&) const {return false;}
};
using Tree=nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double,Adapter>,Adapter,3,size_t>;
struct Correspondence { Eigen::Matrix<double,1,6> J; double r; };
struct System {
  Eigen::Matrix<double,6,6> H=Eigen::Matrix<double,6,6>::Zero();
  Vec6 b=Vec6::Zero(); size_t n=0; double rmse=std::numeric_limits<double>::quiet_NaN(),ratio=0,condition=std::numeric_limits<double>::quiet_NaN(); std::string error;
};
System linearize(const Cloud& source,const NormalCloud& target,const Tree& tree,
                 const Pose& T,const Config& cfg) {
  std::vector<Correspondence> matches; const double max2=cfg.max_correspondence*cfg.max_correspondence;
  for(const auto& s:source) {
    const Eigen::Vector3d q=T*s; size_t idx=0; double dist=0;
    if(tree.knnSearch(q.data(),1,&idx,&dist)!=1 || dist>max2) continue;
    const auto& n=target.normals[idx]; if(n.squaredNorm()<0.9) continue;
    const double r=n.dot(q-target.points[idx]);
    Correspondence c; c.J.head<3>()=q.cross(n).transpose(); c.J.tail<3>()=n.transpose(); c.r=r;
    matches.push_back(c);
  }
  std::stable_sort(matches.begin(),matches.end(),[](const auto& a,const auto& b){return std::abs(a.r)<std::abs(b.r);});
  const size_t keep=static_cast<size_t>(std::floor(cfg.trim_fraction*matches.size())); matches.resize(keep);
  System sys; sys.n=matches.size(); sys.ratio=double(sys.n)/source.size();
  if(sys.n<cfg.min_correspondences || sys.ratio<cfg.min_inlier_ratio) {sys.error="insufficient_correspondences";return sys;}
  double sum=0;
  for(const auto& c:matches) {
    const double weight=std::abs(c.r)<=cfg.huber ? 1.0 : cfg.huber/std::abs(c.r);
    sys.H.noalias()+=weight*c.J.transpose()*c.J; sys.b.noalias()+=weight*c.J.transpose()*c.r; sum+=c.r*c.r;
  }
  sys.rmse=std::sqrt(sum/sys.n);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,6,6>> eig(sys.H/double(sys.n));
  if(eig.info()!=Eigen::Success || !eig.eigenvalues().allFinite()) {sys.error="nonfinite_eigensystem";return sys;}
  const double lo=eig.eigenvalues()[0],hi=eig.eigenvalues()[5];
  sys.condition=lo>0 ? hi/lo : std::numeric_limits<double>::max();
  if(lo<cfg.min_eigenvalue || sys.condition>cfg.max_condition) sys.error="degenerate_normal_equations";
  return sys;
}
}
Cloud readScan(const std::filesystem::path& path,double min_range,double max_range) {
  std::ifstream f(path,std::ios::binary|std::ios::ate); if(!f) throw std::runtime_error("scan_open_failed:"+path.string());
  const std::streamoff bytes=f.tellg(); if(bytes<=0 || bytes%16!=0) throw std::runtime_error("invalid_scan_byte_length");
  // Decode little endian explicitly, independent of host endianness. Input size is capped before allocation.
  if(bytes>16*1000000) throw std::runtime_error("scan_exceeds_one_million_point_guard");
  f.seekg(0); Cloud p; std::array<unsigned char,16> buffer{};
  while(f.read(reinterpret_cast<char*>(buffer.data()),16)) {
    std::array<float,4> values{};
    for(size_t j=0;j<4;++j) {uint32_t u=0;for(size_t k=0;k<4;++k)u|=uint32_t(buffer[j*4+k])<<(8*k);std::memcpy(&values[j],&u,4);}
    for(float v:values) if(!std::isfinite(v)) throw std::runtime_error("nonfinite_scan_point");
    Eigen::Vector3d v(values[0],values[1],values[2]); const double r=v.norm();
    if(r>=min_range && r<=max_range) p.push_back(v);
  }
  if(!f.eof() || p.empty()) throw std::runtime_error("empty_or_truncated_scan"); return p;
}
Cloud voxelFilter(const Cloud& cloud,double leaf,size_t cap) {
  if(!(leaf>0) || cap==0) throw std::runtime_error("invalid_voxel_config");
  struct Cell {Eigen::Vector3d sum=Eigen::Vector3d::Zero(); size_t n=0;};
  std::map<std::array<int64_t,3>,Cell> cells;
  for(const auto& p:cloud) {
    if(!p.allFinite() || p.cwiseAbs().maxCoeff()/leaf>9e15) throw std::runtime_error("invalid_voxel_point");
    std::array<int64_t,3> key{};for(size_t d=0;d<3;++d)key[d]=int64_t(std::floor(p[d]/leaf));
    auto& cell=cells[key];cell.sum+=p;++cell.n;
  }
  Cloud full;for(const auto& kv:cells)full.push_back(kv.second.sum/double(kv.second.n));
  if(full.size()<=cap)return full;
  // Uniform indices in lexicographically ordered occupied cells: deterministic, no random reservoir.
  Cloud limited;limited.reserve(cap);for(size_t i=0;i<cap;++i)limited.push_back(full[i*full.size()/cap]);return limited;
}
NormalCloud estimateNormals(const Cloud& cloud,const Config& cfg) {
  NormalCloud out{cloud,Cloud(cloud.size(),Eigen::Vector3d::Zero())};
  if(cfg.normal_k<3 || !(cfg.normal_radius>0))throw std::runtime_error("invalid_normal_config");
  if(cloud.size()<cfg.normal_k)return out; Adapter data{cloud};Tree tree(3,data,{16});
  std::vector<size_t> ids(cfg.normal_k);std::vector<double> dists(cfg.normal_k);
  for(size_t i=0;i<cloud.size();++i) {
    const size_t n=tree.knnSearch(cloud[i].data(),cfg.normal_k,ids.data(),dists.data());
    if(n<cfg.normal_k || dists.back()>cfg.normal_radius*cfg.normal_radius)continue;
    Eigen::Vector3d mean=Eigen::Vector3d::Zero();for(size_t id:ids)mean+=cloud[id];mean/=double(n);
    Eigen::Matrix3d cov=Eigen::Matrix3d::Zero();for(size_t id:ids){auto d=(cloud[id]-mean).eval();cov.noalias()+=d*d.transpose();}
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(cov/double(n));
    if(eig.info()!=Eigen::Success || eig.eigenvalues()[1]<1e-8 || eig.eigenvalues()[0]/eig.eigenvalues()[1]>cfg.normal_planarity)continue;
    Eigen::Vector3d normal=eig.eigenvectors().col(0);if(normal.dot(cloud[i])>0)normal=-normal;out.normals[i]=normal;
  }
  return out;
}
IcpResult align(const Cloud& source,const NormalCloud& target,const Pose& initial,const Config& cfg) {
  IcpResult result;result.T=initial;
  if(source.empty() || target.points.empty() || target.points.size()!=target.normals.size() || !validPose(initial)) {result.reason="invalid_or_empty_icp_input";return result;}
  for(const auto& p:source)if(!p.allFinite()){result.reason="nonfinite_source";return result;}
  for(size_t i=0;i<target.points.size();++i)if(!target.points[i].allFinite() || !target.normals[i].allFinite()){result.reason="nonfinite_target";return result;}
  // Recenter the target coordinate system before computing conditioning. A
  // kilometer of arbitrary world-origin offset must not look like degeneracy.
  Eigen::Vector3d center=Eigen::Vector3d::Zero();for(const auto& p:target.points)center+=p;center/=double(target.points.size());
  NormalCloud centered=target;for(auto& p:centered.points)p-=center;
  Pose from_center=Pose::Identity();from_center.translation()=center;
  result.T=from_center.inverse()*initial;
  auto finish=[&](){IcpResult out=result;out.T=from_center*result.T;return out;};
  Adapter data{centered.points};Tree tree(3,data,{16});
  for(size_t iter=0;iter<cfg.max_iterations;++iter) {
    const auto s=linearize(source,centered,tree,result.T,cfg);result.iterations=iter+1;
    result.inliers=s.n;result.rmse=s.rmse;result.inlier_ratio=s.ratio;result.condition=s.condition;
    if(!s.error.empty()){result.reason=s.error;return finish();}
    Eigen::LDLT<Eigen::Matrix<double,6,6>> solver(s.H);const Vec6 delta=solver.solve(-s.b);
    if(solver.info()!=Eigen::Success || !delta.allFinite()){result.reason="nonfinite_or_failed_solve";return finish();}
    if(delta.head<3>().norm()>cfg.max_step_rotation || delta.tail<3>().norm()>cfg.max_step_translation){result.reason="step_limit";return finish();}
    result.T=expSE3(delta)*result.T;
    if(delta.head<3>().norm()<cfg.rotation_epsilon && delta.tail<3>().norm()<cfg.translation_epsilon) {
      const auto final=linearize(source,centered,tree,result.T,cfg);result.inliers=final.n;result.rmse=final.rmse;result.inlier_ratio=final.ratio;result.condition=final.condition;
      result.reason=!final.error.empty()?final.error:(final.rmse>cfg.max_rmse?"fitness_gate":"converged");
      result.converged=result.reason=="converged";return finish();
    }
  }
  result.reason="max_iterations";return finish();
}
}  // namespace slam
