#include "slam/core.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <set>
#include <sstream>
#include <stdexcept>

namespace slam {
std::string quote(const std::string& value) {
  std::ostringstream o;o<<'"';for(unsigned char c:value){if(c=='"'||c=='\\')o<<'\\'<<c;else if(c=='\n')o<<"\\n";else if(c=='\r')o<<"\\r";else if(c=='\t')o<<"\\t";else if(c<32)o<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<int(c)<<std::dec;else o<<c;}o<<'"';return o.str();
}
Config readConfig(const std::filesystem::path& file) {
  std::ifstream f(file);if(!f)throw std::runtime_error("config_open_failed");Config c;std::set<std::string> seen;std::string line;
  while(std::getline(f,line)){if(line.empty() || line[0]=='#')continue;const auto pos=line.find('=');if(pos==std::string::npos)throw std::runtime_error("invalid_flat_config_line");
    const auto key=line.substr(0,pos),value=line.substr(pos+1);if(!seen.insert(key).second)throw std::runtime_error("duplicate_config_key");
    size_t used=0;const double n=std::stod(value,&used);if(used!=value.size() || !std::isfinite(n))throw std::runtime_error("invalid_config_number");
#define NUM(name) if(key==#name){c.name=n;continue;}
#define COUNT(name) if(key==#name){if(n<1 || n!=std::floor(n) || n>10000000)throw std::runtime_error("invalid_config_count");c.name=static_cast<size_t>(n);continue;}
    NUM(voxel) NUM(map_voxel) NUM(min_range) NUM(max_range) COUNT(max_scan_points) COUNT(max_map_points)
    COUNT(local_keyframes) COUNT(max_keyframes) COUNT(normal_k) COUNT(min_correspondences)
    NUM(normal_radius) NUM(normal_planarity) NUM(max_correspondence) NUM(huber) NUM(trim_fraction)
    NUM(max_rmse) NUM(min_inlier_ratio) NUM(max_condition) NUM(min_eigenvalue)
    NUM(translation_epsilon) NUM(rotation_epsilon) NUM(max_step_translation) NUM(max_step_rotation) COUNT(max_iterations)
    NUM(keyframe_distance) NUM(keyframe_yaw) COUNT(descriptor_rings) COUNT(descriptor_sectors) COUNT(temporal_exclusion)
    COUNT(loop_candidates) NUM(descriptor_radius) NUM(sensor_height) NUM(descriptor_threshold) NUM(descriptor_yaw_margin)
    NUM(loop_correspondence) NUM(loop_rmse) NUM(loop_inlier_ratio) COUNT(loop_min_correspondences)
    NUM(loop_reciprocal_translation) NUM(loop_reciprocal_rotation) NUM(odometry_rotation_sigma)
    NUM(odometry_translation_sigma) NUM(loop_rotation_sigma) NUM(loop_translation_sigma) NUM(loop_huber)
    COUNT(graph_iterations) COUNT(expected_frames)
#undef NUM
#undef COUNT
    throw std::runtime_error("unknown_config_key:"+key);
  }
  const std::set<std::string> required={"voxel","map_voxel","min_range","max_range","max_scan_points","max_map_points","local_keyframes","max_keyframes","normal_k","min_correspondences","normal_radius","normal_planarity","max_correspondence","huber","trim_fraction","max_rmse","min_inlier_ratio","max_condition","min_eigenvalue","translation_epsilon","rotation_epsilon","max_step_translation","max_step_rotation","max_iterations","keyframe_distance","keyframe_yaw","descriptor_rings","descriptor_sectors","temporal_exclusion","loop_candidates","descriptor_radius","sensor_height","descriptor_threshold","loop_correspondence","loop_rmse","loop_inlier_ratio","loop_min_correspondences","loop_reciprocal_translation","loop_reciprocal_rotation","odometry_rotation_sigma","odometry_translation_sigma","loop_rotation_sigma","loop_translation_sigma","loop_huber","graph_iterations","expected_frames"};
  auto all_required=required;all_required.insert("descriptor_yaw_margin");
  if(seen!=all_required)throw std::runtime_error("flat_config_missing_explicit_parameters");
  validateConfig(c);return c;
}
void validateConfig(const Config& c) {
  for(double x:{c.voxel,c.map_voxel,c.max_range,c.normal_radius,c.max_correspondence,c.huber,c.max_rmse,c.max_condition,c.min_eigenvalue,c.translation_epsilon,c.rotation_epsilon,c.max_step_translation,c.max_step_rotation,c.keyframe_distance,c.keyframe_yaw,c.descriptor_radius,c.sensor_height,c.descriptor_threshold,c.loop_correspondence,c.loop_rmse,c.loop_reciprocal_translation,c.loop_reciprocal_rotation,c.odometry_rotation_sigma,c.odometry_translation_sigma,c.loop_rotation_sigma,c.loop_translation_sigma,c.loop_huber})if(!std::isfinite(x)||x<=0)throw std::runtime_error("nonpositive_config_parameter");
  if(c.min_range<0 || c.min_range>=c.max_range || c.trim_fraction<=0 || c.trim_fraction>1 ||
     c.min_inlier_ratio<=0 || c.min_inlier_ratio>1 || c.loop_inlier_ratio<=0 || c.loop_inlier_ratio>1 ||
     c.normal_planarity<=0 || c.normal_planarity>=1 || c.descriptor_yaw_margin<=0 || c.descriptor_yaw_margin>=1 || c.normal_k<3 || c.descriptor_sectors<6 ||
     c.local_keyframes>c.max_keyframes || c.min_correspondences>c.max_scan_points ||
     c.loop_min_correspondences>c.max_scan_points)throw std::runtime_error("inconsistent_config");
}
Calibration readCalibration(const std::filesystem::path& file) {
  Calibration c;std::ifstream f(file);if(!f)throw std::runtime_error("calibration_open_failed");
  std::set<std::string> found;std::string line;while(std::getline(f,line)) {
    std::istringstream s(line);std::string key;s>>key;if(key!="P2:" && key!="Tr:")continue;
    if(!found.insert(key).second)throw std::runtime_error("duplicate_calibration_key");Eigen::Matrix<double,3,4> M;
    for(int r=0;r<3;++r)for(int col=0;col<4;++col)if(!(s>>M(r,col)) || !std::isfinite(M(r,col)))throw std::runtime_error("invalid_calibration_values");
    std::string extra;if(s>>extra)throw std::runtime_error("extra_calibration_values");
    if(key=="P2:")c.P2=M;else c.Tr.matrix().topRows<3>()=M;
  }
  if(found.size()!=2 || !validPose(c.Tr))throw std::runtime_error("missing_or_invalid_odometry_calibration");return c;
}
std::vector<double> readTimes(const std::filesystem::path& file) {
  std::ifstream f(file);if(!f)throw std::runtime_error("times_open_failed");std::vector<double> out;std::string line;
  while(std::getline(f,line)){std::istringstream s(line);double t;std::string extra;if(!(s>>t) || (s>>extra) || !std::isfinite(t) || t<0 || (!out.empty() && t<=out.back()))throw std::runtime_error("invalid_timestamp");out.push_back(t);}return out;
}
std::vector<std::filesystem::path> scanFiles(const std::filesystem::path& dir,size_t expected) {
  std::vector<std::filesystem::path> out;for(const auto& e:std::filesystem::directory_iterator(dir)){
    if(!e.is_regular_file() || e.path().extension()!=".bin")throw std::runtime_error("unexpected_scan_directory_member");out.push_back(e.path());}
  std::sort(out.begin(),out.end());if(out.size()!=expected)throw std::runtime_error("scan_count_mismatch");
  for(size_t i=0;i<out.size();++i){std::ostringstream name;name<<std::setw(6)<<std::setfill('0')<<i<<".bin";if(out[i].filename()!=name.str())throw std::runtime_error("noncontiguous_scan_ids");}return out;
}
void writePoses(const std::filesystem::path& file,const std::vector<Pose>& poses,const Pose& Tr) {
  std::ofstream f(file);f<<std::setprecision(17);for(const auto& X:poses){const Pose T=cameraRelative(X,Tr);for(int r=0;r<3;++r)for(int c=0;c<4;++c)f<<T.matrix()(r,c)<<((r==2 && c==3)?'\n':' ');}f.flush();if(!f)throw std::runtime_error("pose_write_failed");
}
}  // namespace slam
