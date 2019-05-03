cmake_minimum_required(VERSION 2.8)
project(binarizewolfjolion)
SET(OpenCV_DIR "/afs/cs.cmu.edu/academic/class/15418-s19/public/projects/park-chen/local/share/OpenCV")
find_package( OpenCV REQUIRED PATHS /afs/cs.cmu.edu/academic/class/15418-s19/public/projects/park-chen/local/ NO_DEFAULT_PATH)
# SET(OpenCV_DIR "/afs/cs.cmu.edu/academic/class/15418-s19/public/projects/park-chen/local_baseline/share/OpenCV/")
# FIND_PACKAGE( OpenCV REQUIRED )


# ----------------------------------------------------------------------------
# Add CUDA libraries (needed for apps/tools, samples)
# ----------------------------------------------------------------------------

#SET(CUDA_DIR "/usr/local/depot/cuda-8.0/")
#find_package(CUDA  REQUIRED PATHS /usr/local/depot/cuda-8.0/ NO_DEFAULT_PATH)
find_package(CUDA REQUIRED)
if(NOT HAVE_CUDA)
  set(ENABLE_DYNAMIC_CUDA OFF)
endif()

message("CUDA: ${CUDA_DIR}")
message(STATUS "CUDA library status:")
message(STATUS "    version: ${CUDA_VERSION}")
message(STATUS "    libraries: ${CUDA_CUDA_LIBRARY}")
message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")

if(HAVE_CUDA AND NOT ENABLE_DYNAMIC_CUDA)
  set(OPENCV_LINKER_LIBS ${OPENCV_LINKER_LIBS} ${CUDA_LIBRARIES} ${CUDA_npp_LIBRARY})
  if(HAVE_CUBLAS)
    set(OPENCV_LINKER_LIBS ${OPENCV_LINKER_LIBS} ${CUDA_cublas_LIBRARY})
  endif()
  if(HAVE_CUFFT)
    set(OPENCV_LINKER_LIBS ${OPENCV_LINKER_LIBS} ${CUDA_cufft_LIBRARY})
  endif()
endif()

CUDA_COMPILE(CU_O binarizewolfjolion_sauvola.cu)
add_executable(binarizewolfjolion_sauvola main.cpp timing.cpp ${CU_O})
target_link_libraries( binarizewolfjolion_sauvola ${OpenCV_LIBS} )

CUDA_COMPILE(CU_1 binarizewolfjolion.cu)
add_executable(binarizewolfjolion main.cpp timing.cpp ${CU_1})
target_link_libraries( binarizewolfjolion ${OpenCV_LIBS} )