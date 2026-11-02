#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <array>
#include <filesystem>
#include <map>
#include <limits>
#include <string>
#include <vector>

namespace slam {
using Pose = Eigen::Isometry3d;
using Vec6 = Eigen::Matrix<double,6,1>;
using Cloud = std::vector<Eigen::Vector3d>;
Eigen::Matrix3d skew(const Eigen::Vector3d& v);
Pose expSE3(const Vec6& x);
Vec6 logSE3(const Pose& T);
double angle(const Pose& T);
bool validPose(const Pose& T);
Eigen::Matrix3d replaceHeading(const Eigen::Matrix3d& rotation, double desired_heading);
Pose cameraRelative(const Pose& X, const Pose& Tr);
std::vector<Pose> propagateCorrections(const std::vector<Pose>& odom,
    const std::vector<size_t>& frames, const std::vector<Pose>& optimized,
    const std::vector<double>& times);

struct Config {
  double voxel=0.4, map_voxel=0.5, min_range=2.0, max_range=80.0;
  size_t max_scan_points=35000, max_map_points=100000, local_keyframes=15;
  size_t max_keyframes=4000, normal_k=20, min_correspondences=100;
  double normal_radius=2.0, normal_planarity=0.20;
  double max_correspondence=1.5, huber=0.15, trim_fraction=0.9;
  double max_rmse=0.30, min_inlier_ratio=0.15, max_condition=1e8;
  double min_eigenvalue=1e-6, translation_epsilon=1e-4, rotation_epsilon=1e-5;
  double max_step_translation=1.0, max_step_rotation=0.20;
  size_t max_iterations=40;
  double keyframe_distance=1.5, keyframe_yaw=0.17453292519943295;
  size_t descriptor_rings=20, descriptor_sectors=60, temporal_exclusion=30;
  size_t loop_candidates=5;
  double descriptor_radius=80.0, sensor_height=1.73, descriptor_threshold=0.20, descriptor_yaw_margin=0.01;
  double loop_correspondence=1.0, loop_rmse=0.20, loop_inlier_ratio=0.30;
  size_t loop_min_correspondences=150;
  double loop_reciprocal_translation=0.15, loop_reciprocal_rotation=0.03490658503988659;
  double odometry_rotation_sigma=0.03, odometry_translation_sigma=0.15;
  double loop_rotation_sigma=0.02, loop_translation_sigma=0.08, loop_huber=1.345;
  size_t graph_iterations=100, expected_frames=1101;
};
Config readConfig(const std::filesystem::path& file);
void validateConfig(const Config& c);
Cloud readScan(const std::filesystem::path& path, double min_range, double max_range);
Cloud voxelFilter(const Cloud& cloud, double leaf, size_t cap);
struct NormalCloud { Cloud points; Cloud normals; };
NormalCloud estimateNormals(const Cloud& cloud, const Config& cfg);
struct IcpResult {
  Pose T=Pose::Identity(); bool converged=false; size_t inliers=0, iterations=0;
  double rmse=std::numeric_limits<double>::quiet_NaN(), inlier_ratio=0;
  double condition=std::numeric_limits<double>::quiet_NaN(); std::string reason;
};
IcpResult align(const Cloud& source, const NormalCloud& target, const Pose& initial,
                const Config& cfg);
struct Descriptor { Eigen::MatrixXd bins; Eigen::VectorXd ring_key; };
Descriptor describe(const Cloud& cloud, const Config& cfg);
struct DescriptorMatch { double distance; double yaw; size_t shift; double ambiguity_margin; };
DescriptorMatch compare(const Descriptor& query, const Descriptor& candidate);
struct VerifiedLoop { size_t i, j; Pose T_i_j; IcpResult forward, reverse; };
bool verifyLoop(const Cloud& source, const Cloud& candidate, const Pose& initial,
                const Config& cfg, VerifiedLoop& verified, std::string& reason);
class PoseGraph {
 public:
  PoseGraph(const Config& cfg);
  ~PoseGraph();
  PoseGraph(const PoseGraph&)=delete; PoseGraph& operator=(const PoseGraph&)=delete;
  void addPose(const Pose& X);
  void addOdometry(size_t i,size_t j,const Pose& T_i_j);
  void addLoop(const VerifiedLoop& loop);
  std::vector<Pose> optimize();
 private:
  struct Impl; Impl* impl_;
};
struct Calibration { Pose Tr=Pose::Identity(); Eigen::Matrix<double,3,4> P2; };
Calibration readCalibration(const std::filesystem::path& file);
std::vector<double> readTimes(const std::filesystem::path& file);
std::vector<std::filesystem::path> scanFiles(const std::filesystem::path& dir,
                                           size_t expected);
void writePoses(const std::filesystem::path& file,const std::vector<Pose>& poses,
                const Pose& Tr);
std::string quote(const std::string& value);
int run(const std::filesystem::path& sequence,const std::filesystem::path& config,
        const std::filesystem::path& output);
}  // namespace slam
