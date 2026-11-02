#include "slam/core.hpp"
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/linear/NoiseModel.h>
#include <stdexcept>

namespace slam {
struct PoseGraph::Impl {Config cfg;gtsam::NonlinearFactorGraph graph;gtsam::Values initial;size_t count=0;};
namespace {
gtsam::Pose3 convert(const Pose& T){if(!validPose(T))throw std::runtime_error("invalid_graph_pose");return gtsam::Pose3(T.matrix());}
gtsam::SharedNoiseModel noise(double rot,double trans) {
  gtsam::Vector6 sigma;sigma<<rot,rot,rot,trans,trans,trans;
  return gtsam::noiseModel::Diagonal::Sigmas(sigma);
}
}
PoseGraph::PoseGraph(const Config& cfg):impl_(new Impl{cfg,{},{},0}){}
PoseGraph::~PoseGraph(){delete impl_;}
void PoseGraph::addPose(const Pose& X) {
  const auto i=impl_->count++;impl_->initial.insert(i,convert(X));
  if(i==0)impl_->graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(i,convert(X),noise(1e-6,1e-6));
}
void PoseGraph::addOdometry(size_t i,size_t j,const Pose& T) {
  if(j!=i+1 || j>=impl_->count)throw std::runtime_error("invalid_odometry_factor_indices");
  impl_->graph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(i,j,convert(T),noise(impl_->cfg.odometry_rotation_sigma,impl_->cfg.odometry_translation_sigma));
}
void PoseGraph::addLoop(const VerifiedLoop& loop) {
  if(!loop.forward.converged || !loop.reverse.converged || loop.i>=loop.j || loop.j>=impl_->count)
    throw std::runtime_error("unverified_loop_or_invalid_indices");
  const auto model=gtsam::noiseModel::Robust::Create(gtsam::noiseModel::mEstimator::Huber::Create(impl_->cfg.loop_huber),noise(impl_->cfg.loop_rotation_sigma,impl_->cfg.loop_translation_sigma));
  impl_->graph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(loop.i,loop.j,convert(loop.T_i_j),model);
}
std::vector<Pose> PoseGraph::optimize() {
  if(impl_->count==0)throw std::runtime_error("empty_pose_graph");
  gtsam::LevenbergMarquardtParams params;params.setMaxIterations(impl_->cfg.graph_iterations);params.setRelativeErrorTol(1e-8);
  const auto optimized=gtsam::LevenbergMarquardtOptimizer(impl_->graph,impl_->initial,params).optimize();
  std::vector<Pose> out;for(size_t i=0;i<impl_->count;++i){Pose T=Pose::Identity();T.matrix()=optimized.at<gtsam::Pose3>(i).matrix();if(!validPose(T))throw std::runtime_error("invalid_optimized_pose");out.push_back(T);}return out;
}
}  // namespace slam
