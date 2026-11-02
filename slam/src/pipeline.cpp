#include "slam/core.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <deque>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <stdexcept>

namespace slam {
namespace {
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
struct Keyframe{size_t frame;Pose X;Descriptor descriptor;};
struct Local{size_t key;Cloud cloud;};
Cloud transformed(const Cloud& p,const Pose& X){Cloud out;out.reserve(p.size());for(const auto& v:p)out.push_back(X*v);return out;}
void number(std::ostream& o,double value){if(std::isfinite(value))o<<value;else o<<"null";}
void poseRow(std::ostream& o,const Pose& X,const Pose& Tr){const Pose T=cameraRelative(X,Tr);for(int r=0;r<3;++r)for(int c=0;c<4;++c)o<<T.matrix()(r,c)<<((r==2 && c==3)?'\n':' ');}
void stats(std::ostream& o,const IcpResult& r){o<<"\"converged\":"<<(r.converged?"true":"false")<<",\"inliers\":"<<r.inliers<<",\"rmse_m\":";number(o,r.rmse);o<<",\"inlier_ratio\":"<<r.inlier_ratio<<",\"condition\":";number(o,r.condition);o<<",\"iterations\":"<<r.iterations<<",\"reason\":"<<quote(r.reason);}
}
int run(const std::filesystem::path& sequence,const std::filesystem::path& config,const std::filesystem::path& output) {
  if(!std::filesystem::create_directory(output))throw std::runtime_error("output_exists_or_parent_missing");
  std::ofstream journal(output/"events.jsonl");journal<<std::setprecision(17);
  size_t active_frame=0,completed_frames=0,expected_frames=0;
  double active_timestamp=std::numeric_limits<double>::quiet_NaN();
  try {
    const auto cfg=readConfig(config);expected_frames=cfg.expected_frames;const auto files=scanFiles(sequence/"velodyne",cfg.expected_frames);
    const auto times=readTimes(sequence/"times.txt");if(times.size()!=files.size())throw std::runtime_error("timestamp_count_mismatch");
    const auto calib=readCalibration(sequence/"calib.txt");
    // These durable accepted-prefix files remain available if a later frame or
    // graph solve fails. Never put an unconverged attempted pose into them.
    std::ofstream partial(output/"poses_odom_partial.txt"),partial_times(output/"timestamps_partial.txt");
    partial<<std::setprecision(17);partial_times<<std::setprecision(17);
    std::ofstream timing(output/"frame_times.csv"),looplog(output/"loops.jsonl"),kfout(output/"keyframes.csv");
    timing<<std::setprecision(17)<<"frame,timestamp_s,io_ms,preprocess_ms,map_normals_ms,registration_ms,loop_ms,frame_inclusive_ms,frame_cpu_ms,inliers,rmse_m,condition,iterations,keyframe\n";
    looplog<<std::setprecision(17);kfout<<"keyframe,frame,timestamp_s\n"<<std::setprecision(17);
    PoseGraph graph(cfg);std::vector<Pose> odom;std::vector<size_t> keyframes;std::vector<Keyframe> archive;
    std::deque<Local> local;NormalCloud map;
    const auto all_start=Clock::now();const auto cpu_start=std::clock();double graph_ms=0;size_t accepted=0;
    for(size_t frame=0;frame<files.size();++frame) {
      active_frame=frame;active_timestamp=times[frame];const auto frame_start=Clock::now();const auto frame_cpu_start=std::clock();auto stage=Clock::now();
      const auto raw=readScan(files[frame],cfg.min_range,cfg.max_range);const double io_ms=elapsed(stage);stage=Clock::now();
      const auto scan=voxelFilter(raw,cfg.voxel,cfg.max_scan_points);const double prep_ms=elapsed(stage);double reg_ms=0,map_ms=0,loop_ms=0;
      Pose X=Pose::Identity();IcpResult registration;registration.reason="origin";registration.converged=true;
      if(frame) {
        const Pose guess=frame>1?odom.back()*(odom[frame-2].inverse()*odom.back()):odom.back();
        stage=Clock::now();registration=align(scan,map,guess,cfg);reg_ms=elapsed(stage);
        if(!registration.converged){journal<<"{\"event\":\"registration_failed\",\"frame\":"<<frame<<",\"timestamp_s\":"<<times[frame]<<",\"accepted_prefix_frames\":"<<completed_frames<<",";stats(journal,registration);journal<<"}\n";journal.flush();throw std::runtime_error("registration_failed:"+registration.reason);}
        X=registration.T;
      }
      odom.push_back(X);
      const Pose since=archive.empty()?Pose::Identity():archive.back().X.inverse()*X;
      const bool key=frame==0 || since.translation().norm()>=cfg.keyframe_distance || angle(since)>=cfg.keyframe_yaw;
      if(key) {
        if(archive.size()>=cfg.max_keyframes)throw std::runtime_error("max_keyframes_exceeded");
        const size_t id=archive.size();graph.addPose(X);
        if(id)graph.addOdometry(id-1,id,archive.back().X.inverse()*X);
        const auto desc=describe(scan,cfg);stage=Clock::now();
        std::vector<std::pair<double,size_t>> candidates;
        for(size_t j=0;j<archive.size();++j) {
          const double distance=(desc.ring_key-archive[j].descriptor.ring_key).norm();
          if(frame-archive[j].frame<=cfg.temporal_exclusion) {
            looplog<<"{\"query_keyframe\":"<<id<<",\"candidate_keyframe\":"<<j<<",\"query_frame\":"<<frame<<",\"candidate_frame\":"<<archive[j].frame<<",\"accepted\":false,\"reason\":\"temporal_exclusion\",\"ring_distance\":"<<distance<<"}\n";
          } else candidates.emplace_back(distance,j);
        }
        std::stable_sort(candidates.begin(),candidates.end());if(candidates.size()>cfg.loop_candidates)candidates.resize(cfg.loop_candidates);
        for(const auto& item:candidates) {
          const size_t j=item.second;const auto match=compare(desc,archive[j].descriptor);
          bool accept=false;std::string reason=match.ambiguity_margin<cfg.descriptor_yaw_margin?"descriptor_ambiguous_yaw":"descriptor_gate";VerifiedLoop verified;verified.i=j;verified.j=id;
          if(match.distance<=cfg.descriptor_threshold && match.ambiguity_margin>=cfg.descriptor_yaw_margin) {
            const auto candidate=voxelFilter(readScan(files[archive[j].frame],cfg.min_range,cfg.max_range),cfg.voxel,cfg.max_scan_points);
            Pose initial=archive[j].X.inverse()*X;
            initial.linear()=replaceHeading(initial.linear(),match.yaw);
            accept=verifyLoop(scan,candidate,initial,cfg,verified,reason);
            if(accept){graph.addLoop(verified);++accepted;}
          }
          looplog<<"{\"query_keyframe\":"<<id<<",\"candidate_keyframe\":"<<j<<",\"query_frame\":"<<frame<<",\"candidate_frame\":"<<archive[j].frame<<",\"ring_distance\":"<<item.first<<",\"descriptor_distance\":"<<match.distance<<",\"yaw_rad\":"<<match.yaw<<",\"yaw_ambiguity_margin\":"<<match.ambiguity_margin<<",\"accepted\":"<<(accept?"true":"false")<<",\"reason\":"<<quote(reason)<<",\"forward\":{";stats(looplog,verified.forward);looplog<<"},\"reverse\":{";stats(looplog,verified.reverse);looplog<<"}}\n";
        }
        loop_ms=elapsed(stage);archive.push_back({frame,X,desc});keyframes.push_back(frame);
        kfout<<id<<','<<frame<<','<<times[frame]<<'\n';local.push_back({id,scan});while(local.size()>cfg.local_keyframes)local.pop_front();
        stage=Clock::now();Cloud combined;for(const auto& entry:local){const auto world=transformed(entry.cloud,archive[entry.key].X);combined.insert(combined.end(),world.begin(),world.end());}
        map=estimateNormals(voxelFilter(combined,cfg.map_voxel,cfg.max_map_points),cfg);map_ms=elapsed(stage);
      }
      poseRow(partial,X,calib.Tr);partial_times<<times[frame]<<'\n';partial.flush();partial_times.flush();
      if(!partial || !partial_times)throw std::runtime_error("accepted_prefix_write_failed");
      completed_frames=frame+1;
      timing<<frame<<','<<times[frame]<<','<<io_ms<<','<<prep_ms<<','<<map_ms<<','<<reg_ms<<','<<loop_ms<<','<<elapsed(frame_start)<<','<<1000.0*(std::clock()-frame_cpu_start)/CLOCKS_PER_SEC<<','<<registration.inliers<<','<<registration.rmse<<','<<registration.condition<<','<<registration.iterations<<','<<key<<'\n';
      timing.flush();looplog.flush();journal<<"{\"event\":\"frame_complete\",\"frame\":"<<frame<<",\"monotonic_elapsed_ms\":"<<elapsed(all_start)<<"}\n";journal.flush();
    }
    auto stage=Clock::now();const auto optimized=graph.optimize();graph_ms=elapsed(stage);
    const auto full=propagateCorrections(odom,keyframes,optimized,times);
    writePoses(output/"poses_odom.txt",odom,calib.Tr);writePoses(output/"poses_slam.txt",full,calib.Tr);
    looplog.flush();std::ifstream lines(output/"loops.jsonl");std::ofstream loop_array(output/"loops.json");loop_array<<'[';std::string line;bool first=true;while(std::getline(lines,line)){if(!first)loop_array<<',';loop_array<<'\n'<<line;first=false;}loop_array<<"\n]\n";loop_array.flush();
    std::ofstream stamp(output/"timestamps.txt");stamp<<std::setprecision(17);for(double t:times)stamp<<t<<'\n';
    std::ofstream summary(output/"native_summary.json");summary<<std::setprecision(17)<<"{\"status\":\"complete\",\"frames\":"<<odom.size()<<",\"keyframes\":"<<keyframes.size()<<",\"accepted_loops\":"<<accepted<<",\"graph_ms\":"<<graph_ms<<",\"native_total_ms\":"<<elapsed(all_start)<<",\"process_cpu_ms\":"<<1000.0*(std::clock()-cpu_start)/CLOCKS_PER_SEC<<",\"pose_frame\":\"camera0_initial_relative\",\"internal_frame\":\"lidar_initial_relative\",\"graph_optimizer\":\"end_LM\",\"timestamp_interpolation\":\"correction_SE3_actual_times\"}\n";
    summary.flush();stamp.flush();kfout.flush();
    if(!summary || !timing || !looplog || !loop_array || !kfout || !stamp)throw std::runtime_error("artifact_write_failed");
    journal<<"{\"event\":\"completed\",\"frames\":"<<odom.size()<<"}\n";return 0;
  } catch(const std::exception& error) {
    std::ofstream failure(output/"failure.json");failure<<std::setprecision(17)<<"{\"status\":\"failed\",\"frame\":"<<active_frame<<",\"timestamp_s\":";number(failure,active_timestamp);
    failure<<",\"accepted_prefix_frames\":"<<completed_frames<<",\"expected_frames\":"<<expected_frames<<",\"message\":"<<quote(error.what())<<",\"failed_frame_pose\":null,\"optimized_trajectory\":null}\n";failure.flush();
    journal<<"{\"event\":\"failed\",\"frame\":"<<active_frame<<",\"accepted_prefix_frames\":"<<completed_frames<<",\"message\":"<<quote(error.what())<<"}\n";journal.flush();throw;
  }
}
}  // namespace slam
