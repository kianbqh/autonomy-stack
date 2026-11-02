#include "slam/core.hpp"
#include <cmath>
#include <stdexcept>

namespace slam {
Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
  Eigen::Matrix3d W; W << 0,-v.z(),v.y(),v.z(),0,-v.x(),-v.y(),v.x(),0; return W;
}
Pose expSE3(const Vec6& x) {
  const Eigen::Vector3d w=x.head<3>(); const double a=w.norm();
  const auto W=skew(w); const auto W2=(W*W).eval();
  double A,B,C;
  if(a<1e-6) { const double a2=a*a; A=1-a2/6+a2*a2/120;
    B=0.5-a2/24+a2*a2/720; C=1.0/6-a2/120+a2*a2/5040; }
  else { A=std::sin(a)/a; B=(1-std::cos(a))/(a*a); C=(a-std::sin(a))/(a*a*a); }
  Pose T=Pose::Identity(); T.linear()=Eigen::Matrix3d::Identity()+A*W+B*W2;
  T.translation()=(Eigen::Matrix3d::Identity()+B*W+C*W2)*x.tail<3>(); return T;
}
Vec6 logSE3(const Pose& T) {
  if(!validPose(T)) throw std::runtime_error("invalid SE3 log input");
  Eigen::AngleAxisd aa(T.linear()); Eigen::Vector3d w=aa.axis()*aa.angle();
  const double a=w.norm(); const auto W=skew(w);
  const double c=a<1e-6 ? 1.0/12+a*a/720 : (1-0.5*a/std::tan(0.5*a))/(a*a);
  Vec6 x; x.head<3>()=w;
  x.tail<3>()=(Eigen::Matrix3d::Identity()-0.5*W+c*W*W)*T.translation(); return x;
}
double angle(const Pose& T) { return Eigen::AngleAxisd(T.linear()).angle(); }
bool validPose(const Pose& T) {
  return T.matrix().allFinite() &&
    (T.linear().transpose()*T.linear()-Eigen::Matrix3d::Identity()).norm()<1e-6 &&
    std::abs(T.linear().determinant()-1)<1e-6 &&
    (T.matrix().row(3)-Eigen::RowVector4d(0,0,0,1)).norm()<1e-9;
}
Eigen::Matrix3d replaceHeading(const Eigen::Matrix3d& rotation,double desired_heading) {
  Pose checked=Pose::Identity();checked.linear()=rotation;
  if(!validPose(checked) || !std::isfinite(desired_heading))throw std::runtime_error("invalid_heading_initialization");
  // Left yaw correction preserves the current tilt. Euler decomposition has
  // equivalent branches; replacing only one returned Euler angle is unsafe.
  const double current_heading=std::atan2(rotation(1,0),rotation(0,0));
  return Eigen::AngleAxisd(desired_heading-current_heading,Eigen::Vector3d::UnitZ()).toRotationMatrix()*rotation;
}
Pose cameraRelative(const Pose& X,const Pose& Tr) { return Tr*X*Tr.inverse(); }
std::vector<Pose> propagateCorrections(const std::vector<Pose>& odom,
    const std::vector<size_t>& frames,const std::vector<Pose>& opt,
    const std::vector<double>& times) {
  if(odom.empty() || times.size()!=odom.size() || frames.empty() || frames.front()!=0 ||
     frames.size()!=opt.size()) throw std::runtime_error("invalid correction inputs");
  std::vector<Pose> corrections;
  for(size_t k=0;k<frames.size();++k) {
    if(frames[k]>=odom.size() || (k && frames[k]<=frames[k-1]))
      throw std::runtime_error("unsorted correction keyframes");
    corrections.push_back(opt[k]*odom[frames[k]].inverse());
  }
  std::vector<Pose> result; size_t k=0;
  for(size_t i=0;i<odom.size();++i) {
    if(i && !(times[i]>times[i-1])) throw std::runtime_error("timestamps not strictly increasing");
    while(k+1<frames.size() && i>=frames[k+1]) ++k;
    Pose C=corrections[k];
    if(k+1<frames.size()) {
      const double dt=times[frames[k+1]]-times[frames[k]];
      if(!(dt>0)) throw std::runtime_error("invalid correction timestamp interval");
      const double alpha=(times[i]-times[frames[k]])/dt;
      C=C*expSE3(alpha*logSE3(C.inverse()*corrections[k+1]));
    }
    result.push_back(C*odom[i]);
  }
  return result;
}
}  // namespace slam
