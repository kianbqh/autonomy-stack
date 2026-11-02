#include "slam/core.hpp"
#include <iostream>
int main(int argc,char** argv){if(argc!=4){std::cerr<<"usage: run_kitti_slam SEQUENCE_DIR FLAT_CONFIG NEW_OUTPUT_DIR\n";return 2;}try{return slam::run(argv[1],argv[2],argv[3]);}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
