#include "slam/core.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace slam {
Descriptor describe(const Cloud& cloud,const Config& cfg) {
  Descriptor d;d.bins=Eigen::MatrixXd::Zero(cfg.descriptor_rings,cfg.descriptor_sectors);
  const double tau=2*std::acos(-1.0);
  for(const auto& p:cloud) {
    const double radius=p.head<2>().norm();if(radius>=cfg.descriptor_radius || radius<1e-6)continue;
    double yaw=std::atan2(p.y(),p.x());if(yaw<0)yaw+=tau;
    const size_t r=std::min(cfg.descriptor_rings-1,size_t(radius/cfg.descriptor_radius*cfg.descriptor_rings));
    const size_t s=std::min(cfg.descriptor_sectors-1,size_t(yaw/tau*cfg.descriptor_sectors));
    d.bins(r,s)=std::max(d.bins(r,s),std::max(0.0,p.z()+cfg.sensor_height));
  }
  d.ring_key=d.bins.rowwise().mean();return d;
}
DescriptorMatch compare(const Descriptor& query,const Descriptor& candidate) {
  if(query.bins.rows()!=candidate.bins.rows() || query.bins.cols()!=candidate.bins.cols() || query.bins.size()==0)
    throw std::runtime_error("descriptor_shape_mismatch");
  const size_t sectors=query.bins.cols();DescriptorMatch best{std::numeric_limits<double>::max(),0,0,0};
  std::vector<double> distances;
  for(size_t shift=0;shift<sectors;++shift) {
    double sum=0;size_t count=0;
    for(size_t s=0;s<sectors;++s) {
      const auto q=query.bins.col((s+sectors-shift)%sectors),c=candidate.bins.col(s);
      const double den=q.norm()*c.norm();if(den<1e-10)continue;
      sum+=std::clamp(q.dot(c)/den,-1.0,1.0);++count;
    }
    // Little shared angular support must not earn a good score.
    const double distance=count<sectors/3?1.0:1-sum/double(count);
    distances.push_back(distance);
    if(distance<best.distance)best={distance,2*std::acos(-1.0)*shift/sectors,shift,0};
  }
  double second=std::numeric_limits<double>::max();
  for(size_t shift=0;shift<sectors;++shift){const size_t gap=shift>best.shift?shift-best.shift:best.shift-shift;if(std::min(gap,sectors-gap)>=3)second=std::min(second,distances[shift]);}
  best.ambiguity_margin=second-best.distance;
  return best;
}
bool verifyLoop(const Cloud& source,const Cloud& candidate,const Pose& initial,
                const Config& cfg,VerifiedLoop& loop,std::string& reason) {
  Config gate=cfg;gate.max_correspondence=cfg.loop_correspondence;gate.max_rmse=cfg.loop_rmse;
  gate.min_inlier_ratio=cfg.loop_inlier_ratio;gate.min_correspondences=cfg.loop_min_correspondences;
  loop.forward=align(source,estimateNormals(candidate,gate),initial,gate);
  if(!loop.forward.converged){reason="forward_"+loop.forward.reason;return false;}
  loop.reverse=align(candidate,estimateNormals(source,gate),loop.forward.T.inverse(),gate);
  if(!loop.reverse.converged){reason="reverse_"+loop.reverse.reason;return false;}
  const Pose cycle=loop.forward.T*loop.reverse.T;
  if(cycle.translation().norm()>cfg.loop_reciprocal_translation || angle(cycle)>cfg.loop_reciprocal_rotation){reason="reciprocal_gate";return false;}
  loop.T_i_j=loop.forward.T;reason="accepted";return true;
}
}  // namespace slam
