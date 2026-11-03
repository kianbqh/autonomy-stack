#include "slam/core.hpp"
#include <gtest/gtest.h>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>

namespace {
slam::Cloud room() {
  slam::Cloud c;
  for(int i=-20;i<=20;++i)for(int j=-20;j<=20;++j) {
    const double a=i*.12,b=j*.12;
    c.emplace_back(5,a,b);c.emplace_back(a,-4,b);c.emplace_back(a,b,-2);
  }return c;
}
slam::Config fixtureConfig() {
  slam::Config c;c.normal_k=12;c.normal_radius=.5;c.normal_planarity=.05;
  c.min_correspondences=100;c.max_correspondence=.4;c.huber=.03;c.trim_fraction=.9;
  c.min_inlier_ratio=.5;c.max_rmse=.01;c.translation_epsilon=1e-7;c.rotation_epsilon=1e-8;
  c.max_iterations=80;c.loop_correspondence=.4;c.loop_rmse=.01;c.loop_inlier_ratio=.5;
  return c;
}
}
TEST(Geometry,LeftJacobianFiniteDifference) {
  const Eigen::Vector3d p(2,-1,3),n=Eigen::Vector3d(1,2,3).normalized();
  Eigen::Matrix<double,1,6> J;J.head<3>()=p.cross(n).transpose();J.tail<3>()=n.transpose();
  for(int k=0;k<6;++k){slam::Vec6 d=slam::Vec6::Zero();d[k]=1e-7;EXPECT_NEAR(n.dot(slam::expSE3(d)*p-p)/1e-7,J[k],1e-6);}
}
TEST(Geometry,NontrivialExpLogAndInverse) {
  slam::Vec6 x;x<<.12,-.08,.21,.4,-.2,.7;const auto T=slam::expSE3(x);
  EXPECT_LT((slam::logSE3(T)-x).norm(),1e-10);EXPECT_LT(((T*T.inverse()).matrix()-slam::Pose::Identity().matrix()).norm(),1e-10);
}
TEST(Geometry,HeadingReplacementAvoidsNegativeYawEulerBranch) {
  const double old_yaw=-.4,pitch=.15,roll=-.2,desired=.7;
  const Eigen::Matrix3d R=(Eigen::AngleAxisd(old_yaw,Eigen::Vector3d::UnitZ())*Eigen::AngleAxisd(pitch,Eigen::Vector3d::UnitY())*Eigen::AngleAxisd(roll,Eigen::Vector3d::UnitX())).toRotationMatrix();
  const Eigen::Matrix3d expected=(Eigen::AngleAxisd(desired,Eigen::Vector3d::UnitZ())*Eigen::AngleAxisd(pitch,Eigen::Vector3d::UnitY())*Eigen::AngleAxisd(roll,Eigen::Vector3d::UnitX())).toRotationMatrix();
  const auto branch=R.eulerAngles(2,1,0);
  const Eigen::Matrix3d old_initialization=(Eigen::AngleAxisd(desired,Eigen::Vector3d::UnitZ())*Eigen::AngleAxisd(branch[1],Eigen::Vector3d::UnitY())*Eigen::AngleAxisd(branch[2],Eigen::Vector3d::UnitX())).toRotationMatrix();
  // Eigen3.4 documents a nonnegative first-angle range. This explicitly
  // exercises the formerly unsafe branch on the actual pinned library later.
  EXPECT_GT((old_initialization-expected).norm(),1.0);
  EXPECT_LT((slam::replaceHeading(R,desired)-expected).norm(),1e-12);
}
TEST(Geometry,HeadingReplacementPreservesTiltAcrossPiBoundary) {
  const double pi=std::acos(-1.0);
  const std::array<std::array<double,4>,5> cases={{{-.7,.2,-.15,.4},{-2.9,-.3,.25,2.7},{pi-.01,.25,.3,-pi+.02},{-pi+.01,-.2,-.25,pi-.02},{-.2,0,0,2*pi-.3}}};
  for(const auto& c:cases) {
    const double yaw=c[0],roll=c[1],pitch=c[2],desired=c[3];
    const Eigen::Matrix3d R=(Eigen::AngleAxisd(yaw,Eigen::Vector3d::UnitZ())*Eigen::AngleAxisd(pitch,Eigen::Vector3d::UnitY())*Eigen::AngleAxisd(roll,Eigen::Vector3d::UnitX())).toRotationMatrix();
    const Eigen::Matrix3d expected=(Eigen::AngleAxisd(desired,Eigen::Vector3d::UnitZ())*Eigen::AngleAxisd(pitch,Eigen::Vector3d::UnitY())*Eigen::AngleAxisd(roll,Eigen::Vector3d::UnitX())).toRotationMatrix();
    const Eigen::Matrix3d result=slam::replaceHeading(R,desired);
    EXPECT_LT((result-expected).norm(),1e-12);
    EXPECT_LT((result.row(2)-R.row(2)).norm(),1e-12);
    EXPECT_LT((result.transpose()*result-Eigen::Matrix3d::Identity()).norm(),1e-12);
    EXPECT_NEAR(result.determinant(),1,1e-12);
  }
}
TEST(Calibration,RectifiedConjugationAndGroundTruthOrigin) {
  slam::Vec6 a,b;a<<.2,-.1,.3,.5,-.2,.9;b<<-.12,.03,.08,1.2,.7,-.4;
  const auto Tr=slam::expSE3(a),X=slam::expSE3(b),C=slam::cameraRelative(X,Tr);
  Eigen::Vector3d p(.3,1,4);
  EXPECT_LT((C*(Tr*p)-Tr*(X*p)).norm(),1e-10);
  EXPECT_GT((C.matrix()-X.matrix()).norm(),.1);
  const auto origin=slam::expSE3(a*2);const auto gt_i=origin*C;
  EXPECT_LT(((origin.inverse()*gt_i).matrix()-C.matrix()).norm(),1e-10);
  EXPECT_LT(((Tr.inverse()*C*Tr).matrix()-X.matrix()).norm(),1e-10);
  Eigen::Matrix<double,3,4> P;P<<500,0,300,-50,0,500,200,0,0,0,1,0;
  const Eigen::Vector4d h(p.x(),p.y(),p.z(),1);const auto projected=P*Tr.matrix()*h;
  const auto pc=Tr*p;EXPECT_NEAR(projected[0]/projected[2],(500*pc.x()+300*pc.z()-50)/pc.z(),1e-10);
}
TEST(ICP,KnownSE3RecoveryWithOwnNormals) {
  const auto target=room();const auto cfg=fixtureConfig();const auto normals=slam::estimateNormals(target,cfg);
  slam::Vec6 twist;twist<<.012,-.008,.01,.08,-.06,.05;const auto truth=slam::expSE3(twist);
  slam::Cloud source;for(const auto& p:target)source.push_back(truth.inverse()*p);
  const auto r=slam::align(source,normals,slam::Pose::Identity(),cfg);
  ASSERT_TRUE(r.converged)<<r.reason;const auto error=r.T.inverse()*truth;
  EXPECT_LT(error.translation().norm(),.001);EXPECT_LT(slam::angle(error),.05*std::acos(-1)/180);
}
TEST(ICP,NoisePartialOverlapAndOutliers) {
  const auto target=room();auto cfg=fixtureConfig();const auto normals=slam::estimateNormals(target,cfg);
  slam::Vec6 twist;twist<<.01,-.006,.008,.06,-.04,.03;const auto truth=slam::expSE3(twist);
  std::mt19937 rng(7);std::normal_distribution<double> noise(0,.0002);slam::Cloud source;
  for(size_t i=0;i<target.size();++i)if(i%5){Eigen::Vector3d p=truth.inverse()*target[i];for(int k=0;k<3;++k)p[k]+=noise(rng);source.push_back(p);}
  for(int i=0;i<100;++i)source.emplace_back(30+i*.1,40,50);
  const auto r=slam::align(source,normals,slam::Pose::Identity(),cfg);ASSERT_TRUE(r.converged)<<r.reason;
  EXPECT_LT((r.T.inverse()*truth).translation().norm(),.001);EXPECT_LT(slam::angle(r.T.inverse()*truth),.05*std::acos(-1)/180);
}
TEST(ICP,ConditioningDoesNotDependOnWorldOriginTranslation) {
  auto target=room();const auto cfg=fixtureConfig();slam::Vec6 d;d<<.01,-.006,.008,.06,-.04,.03;
  const auto truth=slam::expSE3(d);slam::Cloud source;for(const auto& p:target)source.push_back(truth.inverse()*p);
  auto offset=slam::Pose::Identity();offset.translation()=Eigen::Vector3d(1000,-2000,300);
  for(auto& p:target)p=offset*p;
  const auto r=slam::align(source,slam::estimateNormals(target,cfg),offset,cfg);
  ASSERT_TRUE(r.converged)<<r.reason;const auto error=r.T.inverse()*offset*truth;
  EXPECT_LT(error.translation().norm(),.001);EXPECT_LT(slam::angle(error),.05*std::acos(-1)/180);
}
TEST(ICP,PlaneDegeneracyAndEmptyInputsFail) {
  slam::Cloud plane;for(int x=-20;x<=20;++x)for(int y=-20;y<=20;++y)plane.emplace_back(x*.1,y*.1,0);
  const auto c=fixtureConfig();const auto r=slam::align(plane,slam::estimateNormals(plane,c),slam::Pose::Identity(),c);
  EXPECT_FALSE(r.converged);EXPECT_EQ(r.reason,"degenerate_normal_equations");
  EXPECT_FALSE(slam::align({}, {},slam::Pose::Identity(),c).converged);
  auto bad=room();bad[0][0]=std::numeric_limits<double>::quiet_NaN();EXPECT_FALSE(slam::align(bad,slam::estimateNormals(plane,c),slam::Pose::Identity(),c).converged);
}
TEST(Descriptor,YawHasCurrentToCandidateSign) {
  slam::Descriptor q,c;q.bins=Eigen::MatrixXd::Zero(2,12);q.bins.col(1)<<1,2;q.bins.col(3)<<2,1;q.bins.col(6)<<3,2;q.bins.col(8)<<1,4;
  c.bins=Eigen::MatrixXd::Zero(2,12);for(int j=0;j<12;++j)c.bins.col((j+2)%12)=q.bins.col(j);
  const auto m=slam::compare(q,c);EXPECT_EQ(m.shift,2);EXPECT_NEAR(m.yaw,std::acos(-1)/3,1e-10);EXPECT_NEAR(m.distance,0,1e-12);
}
TEST(PoseGraph,GeometricallyVerifiedLoopReducesControlledToyDrift) {
  auto c=fixtureConfig();c.loop_translation_sigma=.02;c.loop_rotation_sigma=.01;
  const auto cloud=room();slam::VerifiedLoop loop;std::string reason;
  ASSERT_TRUE(slam::verifyLoop(cloud,cloud,slam::Pose::Identity(),c,loop,reason))<<reason;
  std::vector<Eigen::Vector3d> true_t={{0,0,0},{2,0,0},{2,2,0},{0,2,0},{0,0,0}};
  std::vector<slam::Pose> odom;slam::PoseGraph graph(c);
  for(size_t i=0;i<true_t.size();++i){auto p=slam::Pose::Identity();p.translation()=true_t[i]+Eigen::Vector3d(.1*i,0,0);odom.push_back(p);graph.addPose(p);if(i)graph.addOdometry(i-1,i,odom[i-1].inverse()*p);}
  loop.i=0;loop.j=4;graph.addLoop(loop);const auto opt=graph.optimize();
  EXPECT_LT(opt.back().translation().norm(),odom.back().translation().norm()*.5);
  EXPECT_LT(opt.front().translation().norm(),1e-5);
  slam::VerifiedLoop bad;bad.i=0;bad.j=4;EXPECT_THROW(graph.addLoop(bad),std::runtime_error);
}
TEST(Trajectory,CorrectionUsesActualTimestampAndPreservesAllFrames) {
  std::vector<slam::Pose> odom(3,slam::Pose::Identity()),opt(2,slam::Pose::Identity());
  odom[1].translation().x()=1;odom[2].translation().x()=2;opt[1].translation().x()=4;
  const auto out=slam::propagateCorrections(odom,{0,2},opt,{0,.25,1});
  ASSERT_EQ(out.size(),3);EXPECT_NEAR(out[1].translation().x(),1.5,1e-10);EXPECT_NEAR(out[2].translation().x(),4,1e-10);
}
TEST(IO,ThreeFrameBinaryAndMalformedScan) {
  const auto root=std::filesystem::temp_directory_path()/("slam-test-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  std::filesystem::create_directories(root);
  for(int i=0;i<3;++i){std::ostringstream name;name<<std::setw(6)<<std::setfill('0')<<i<<".bin";std::ofstream f(root/name.str(),std::ios::binary);const std::array<unsigned char,16> p={0,0,128,63,0,0,0,64,0,0,64,64,0,0,0,0};f.write(reinterpret_cast<const char*>(p.data()),p.size());}
  const auto files=slam::scanFiles(root,3);ASSERT_EQ(files.size(),3);EXPECT_NEAR(slam::readScan(files[0],0,80)[0].norm(),std::sqrt(14),1e-8);
  {std::ofstream f(files[1],std::ios::binary|std::ios::app);f.put('x');}EXPECT_THROW(slam::readScan(files[1],0,80),std::runtime_error);
  std::filesystem::remove_all(root);
}

TEST(ICP,InternalRotationRetractionPreventsLongSequenceDrift) {
  const auto target=room();const auto cfg=fixtureConfig();
  const auto normals=slam::estimateNormals(target,cfg);
  // A legitimate near-SO(3) input models floating-point composition error.
  // Exercise production align(), not a standalone normalization helper.
  auto seed=slam::Pose::Identity();seed.linear()(0,0)+=1e-12;
  ASSERT_TRUE(slam::validPose(seed));
  std::vector<slam::Pose> accepted{seed};
  slam::Vec6 motion;motion<<.002,-.003,.004,.01,-.02,.02;
  for(size_t frame=1;frame<=128;++frame) {
    const auto truth=slam::expSE3(double(frame)*motion);
    slam::Cloud source;source.reserve(target.size());
    for(const auto& point:target)source.push_back(truth.inverse()*point);
    const slam::Pose guess=frame>1?accepted.back()*(accepted[frame-2].inverse()*accepted.back()):accepted.back();
    const auto result=slam::align(source,normals,guess,cfg);
    ASSERT_TRUE(result.converged)<<"frame "<<frame<<": "<<result.reason;
    EXPECT_TRUE(slam::validPose(result.T));
    EXPECT_LT((result.T.linear().transpose()*result.T.linear()-Eigen::Matrix3d::Identity()).norm(),1e-12);
    EXPECT_NEAR(result.T.linear().determinant(),1.0,1e-12);
    EXPECT_LT((result.T.inverse()*truth).translation().norm(),.001);
    EXPECT_LT(slam::angle(result.T.inverse()*truth),.05*std::acos(-1)/180);
    accepted.push_back(result.T);
  }
  // Internal retraction must never repair an explicitly invalid caller input.
  auto invalid=slam::Pose::Identity();invalid.linear()(0,0)=1.01;
  ASSERT_FALSE(slam::validPose(invalid));
  const auto refused=slam::align(target,normals,invalid,cfg);
  EXPECT_FALSE(refused.converged);
  EXPECT_EQ(refused.reason,"invalid_or_empty_icp_input");
  EXPECT_EQ((refused.T.matrix()-invalid.matrix()).norm(),0.0);
}
