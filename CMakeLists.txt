cmake_minimum_required(VERSION 2.8)
project(binarizewolfjolion)
enable_language(CUDA)
SET(OpenCV_DIR "/afs/cs.cmu.edu/academic/class/15418-s19/public/projects/park-chen/local/share/OpenCV")
find_package( OpenCV REQUIRED PATHS /afs/cs.cmu.edu/academic/class/15418-s19/public/projects/park-chen/local/ NO_DEFAULT_PATH)

find_package(CUDA QUIET REQUIRED)

# Pass options to NVCC
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 -gencode arch=compute_22,code=sm_22
    )

add_executable(binarizewolfjolion binarizewolfjolion.cu timing.cpp)
target_link_libraries( binarizewolfjolion ${OpenCV_LIBS} )