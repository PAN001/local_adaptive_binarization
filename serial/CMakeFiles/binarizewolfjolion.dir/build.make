# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/ccmake

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /afs/cs.cmu.edu/academic/class/15418-s19/public/projects/park-chen/local_adaptive_binarization/serial

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /afs/cs.cmu.edu/academic/class/15418-s19/public/projects/park-chen/local_adaptive_binarization/serial

# Include any dependencies generated for this target.
include CMakeFiles/binarizewolfjolion.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/binarizewolfjolion.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/binarizewolfjolion.dir/flags.make

CMakeFiles/binarizewolfjolion.dir/binarizewolfjolion.cpp.o: CMakeFiles/binarizewolfjolion.dir/flags.make
CMakeFiles/binarizewolfjolion.dir/binarizewolfjolion.cpp.o: binarizewolfjolion.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /afs/cs.cmu.edu/academic/class/15418-s19/public/projects/park-chen/local_adaptive_binarization/serial/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/binarizewolfjolion.dir/binarizewolfjolion.cpp.o"
	/usr/lib64/ccache/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/binarizewolfjolion.dir/binarizewolfjolion.cpp.o -c /afs/cs.cmu.edu/academic/class/15418-s19/public/projects/park-chen/local_adaptive_binarization/serial/binarizewolfjolion.cpp

CMakeFiles/binarizewolfjolion.dir/binarizewolfjolion.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/binarizewolfjolion.dir/binarizewolfjolion.cpp.i"
	/usr/lib64/ccache/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /afs/cs.cmu.edu/academic/class/15418-s19/public/projects/park-chen/local_adaptive_binarization/serial/binarizewolfjolion.cpp > CMakeFiles/binarizewolfjolion.dir/binarizewolfjolion.cpp.i

CMakeFiles/binarizewolfjolion.dir/binarizewolfjolion.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/binarizewolfjolion.dir/binarizewolfjolion.cpp.s"
	/usr/lib64/ccache/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /afs/cs.cmu.edu/academic/class/15418-s19/public/projects/park-chen/local_adaptive_binarization/serial/binarizewolfjolion.cpp -o CMakeFiles/binarizewolfjolion.dir/binarizewolfjolion.cpp.s

CMakeFiles/binarizewolfjolion.dir/binarizewolfjolion.cpp.o.requires:
.PHONY : CMakeFiles/binarizewolfjolion.dir/binarizewolfjolion.cpp.o.requires

CMakeFiles/binarizewolfjolion.dir/binarizewolfjolion.cpp.o.provides: CMakeFiles/binarizewolfjolion.dir/binarizewolfjolion.cpp.o.requires
	$(MAKE) -f CMakeFiles/binarizewolfjolion.dir/build.make CMakeFiles/binarizewolfjolion.dir/binarizewolfjolion.cpp.o.provides.build
.PHONY : CMakeFiles/binarizewolfjolion.dir/binarizewolfjolion.cpp.o.provides

CMakeFiles/binarizewolfjolion.dir/binarizewolfjolion.cpp.o.provides.build: CMakeFiles/binarizewolfjolion.dir/binarizewolfjolion.cpp.o

CMakeFiles/binarizewolfjolion.dir/timing.cpp.o: CMakeFiles/binarizewolfjolion.dir/flags.make
CMakeFiles/binarizewolfjolion.dir/timing.cpp.o: timing.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /afs/cs.cmu.edu/academic/class/15418-s19/public/projects/park-chen/local_adaptive_binarization/serial/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/binarizewolfjolion.dir/timing.cpp.o"
	/usr/lib64/ccache/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/binarizewolfjolion.dir/timing.cpp.o -c /afs/cs.cmu.edu/academic/class/15418-s19/public/projects/park-chen/local_adaptive_binarization/serial/timing.cpp

CMakeFiles/binarizewolfjolion.dir/timing.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/binarizewolfjolion.dir/timing.cpp.i"
	/usr/lib64/ccache/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /afs/cs.cmu.edu/academic/class/15418-s19/public/projects/park-chen/local_adaptive_binarization/serial/timing.cpp > CMakeFiles/binarizewolfjolion.dir/timing.cpp.i

CMakeFiles/binarizewolfjolion.dir/timing.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/binarizewolfjolion.dir/timing.cpp.s"
	/usr/lib64/ccache/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /afs/cs.cmu.edu/academic/class/15418-s19/public/projects/park-chen/local_adaptive_binarization/serial/timing.cpp -o CMakeFiles/binarizewolfjolion.dir/timing.cpp.s

CMakeFiles/binarizewolfjolion.dir/timing.cpp.o.requires:
.PHONY : CMakeFiles/binarizewolfjolion.dir/timing.cpp.o.requires

CMakeFiles/binarizewolfjolion.dir/timing.cpp.o.provides: CMakeFiles/binarizewolfjolion.dir/timing.cpp.o.requires
	$(MAKE) -f CMakeFiles/binarizewolfjolion.dir/build.make CMakeFiles/binarizewolfjolion.dir/timing.cpp.o.provides.build
.PHONY : CMakeFiles/binarizewolfjolion.dir/timing.cpp.o.provides

CMakeFiles/binarizewolfjolion.dir/timing.cpp.o.provides.build: CMakeFiles/binarizewolfjolion.dir/timing.cpp.o

# Object files for target binarizewolfjolion
binarizewolfjolion_OBJECTS = \
"CMakeFiles/binarizewolfjolion.dir/binarizewolfjolion.cpp.o" \
"CMakeFiles/binarizewolfjolion.dir/timing.cpp.o"

# External object files for target binarizewolfjolion
binarizewolfjolion_EXTERNAL_OBJECTS =

binarizewolfjolion: CMakeFiles/binarizewolfjolion.dir/binarizewolfjolion.cpp.o
binarizewolfjolion: CMakeFiles/binarizewolfjolion.dir/timing.cpp.o
binarizewolfjolion: CMakeFiles/binarizewolfjolion.dir/build.make
binarizewolfjolion: /afs/cs.cmu.edu/academic/class/15418-s19/public/projects/park-chen/local/lib/libopencv_videostab.so.2.4.9
binarizewolfjolion: /afs/cs.cmu.edu/academic/class/15418-s19/public/projects/park-chen/local/lib/libopencv_video.so.2.4.9
binarizewolfjolion: /afs/cs.cmu.edu/academic/class/15418-s19/public/projects/park-chen/local/lib/libopencv_ts.a
binarizewolfjolion: /afs/cs.cmu.edu/academic/class/15418-s19/public/projects/park-chen/local/lib/libopencv_superres.so.2.4.9
binarizewolfjolion: /afs/cs.cmu.edu/academic/class/15418-s19/public/projects/park-chen/local/lib/libopencv_stitching.so.2.4.9
binarizewolfjolion: /afs/cs.cmu.edu/academic/class/15418-s19/public/projects/park-chen/local/lib/libopencv_photo.so.2.4.9
binarizewolfjolion: /afs/cs.cmu.edu/academic/class/15418-s19/public/projects/park-chen/local/lib/libopencv_ocl.so.2.4.9
binarizewolfjolion: /afs/cs.cmu.edu/academic/class/15418-s19/public/projects/park-chen/local/lib/libopencv_objdetect.so.2.4.9
binarizewolfjolion: /afs/cs.cmu.edu/academic/class/15418-s19/public/projects/park-chen/local/lib/libopencv_nonfree.so.2.4.9
binarizewolfjolion: /afs/cs.cmu.edu/academic/class/15418-s19/public/projects/park-chen/local/lib/libopencv_ml.so.2.4.9
binarizewolfjolion: /afs/cs.cmu.edu/academic/class/15418-s19/public/projects/park-chen/local/lib/libopencv_legacy.so.2.4.9
binarizewolfjolion: /afs/cs.cmu.edu/academic/class/15418-s19/public/projects/park-chen/local/lib/libopencv_imgproc.so.2.4.9
binarizewolfjolion: /afs/cs.cmu.edu/academic/class/15418-s19/public/projects/park-chen/local/lib/libopencv_highgui.so.2.4.9
binarizewolfjolion: /afs/cs.cmu.edu/academic/class/15418-s19/public/projects/park-chen/local/lib/libopencv_gpu.so.2.4.9
binarizewolfjolion: /afs/cs.cmu.edu/academic/class/15418-s19/public/projects/park-chen/local/lib/libopencv_flann.so.2.4.9
binarizewolfjolion: /afs/cs.cmu.edu/academic/class/15418-s19/public/projects/park-chen/local/lib/libopencv_features2d.so.2.4.9
binarizewolfjolion: /afs/cs.cmu.edu/academic/class/15418-s19/public/projects/park-chen/local/lib/libopencv_core.so.2.4.9
binarizewolfjolion: /afs/cs.cmu.edu/academic/class/15418-s19/public/projects/park-chen/local/lib/libopencv_contrib.so.2.4.9
binarizewolfjolion: /afs/cs.cmu.edu/academic/class/15418-s19/public/projects/park-chen/local/lib/libopencv_calib3d.so.2.4.9
binarizewolfjolion: /lib64/libGLU.so
binarizewolfjolion: /lib64/libGL.so
binarizewolfjolion: /lib64/libSM.so
binarizewolfjolion: /lib64/libICE.so
binarizewolfjolion: /lib64/libX11.so
binarizewolfjolion: /lib64/libXext.so
binarizewolfjolion: /usr/local/depot/cuda-8.0/lib64/libcudart.so
binarizewolfjolion: /usr/local/depot/cuda-8.0/lib64/libnppc.so
binarizewolfjolion: /usr/local/depot/cuda-8.0/lib64/libnppi.so
binarizewolfjolion: /usr/local/depot/cuda-8.0/lib64/libnpps.so
binarizewolfjolion: /afs/cs.cmu.edu/academic/class/15418-s19/public/projects/park-chen/local/lib/libopencv_nonfree.so.2.4.9
binarizewolfjolion: /afs/cs.cmu.edu/academic/class/15418-s19/public/projects/park-chen/local/lib/libopencv_ocl.so.2.4.9
binarizewolfjolion: /afs/cs.cmu.edu/academic/class/15418-s19/public/projects/park-chen/local/lib/libopencv_gpu.so.2.4.9
binarizewolfjolion: /afs/cs.cmu.edu/academic/class/15418-s19/public/projects/park-chen/local/lib/libopencv_photo.so.2.4.9
binarizewolfjolion: /afs/cs.cmu.edu/academic/class/15418-s19/public/projects/park-chen/local/lib/libopencv_objdetect.so.2.4.9
binarizewolfjolion: /afs/cs.cmu.edu/academic/class/15418-s19/public/projects/park-chen/local/lib/libopencv_legacy.so.2.4.9
binarizewolfjolion: /afs/cs.cmu.edu/academic/class/15418-s19/public/projects/park-chen/local/lib/libopencv_video.so.2.4.9
binarizewolfjolion: /afs/cs.cmu.edu/academic/class/15418-s19/public/projects/park-chen/local/lib/libopencv_ml.so.2.4.9
binarizewolfjolion: /afs/cs.cmu.edu/academic/class/15418-s19/public/projects/park-chen/local/lib/libopencv_calib3d.so.2.4.9
binarizewolfjolion: /afs/cs.cmu.edu/academic/class/15418-s19/public/projects/park-chen/local/lib/libopencv_features2d.so.2.4.9
binarizewolfjolion: /afs/cs.cmu.edu/academic/class/15418-s19/public/projects/park-chen/local/lib/libopencv_highgui.so.2.4.9
binarizewolfjolion: /afs/cs.cmu.edu/academic/class/15418-s19/public/projects/park-chen/local/lib/libopencv_imgproc.so.2.4.9
binarizewolfjolion: /afs/cs.cmu.edu/academic/class/15418-s19/public/projects/park-chen/local/lib/libopencv_flann.so.2.4.9
binarizewolfjolion: /afs/cs.cmu.edu/academic/class/15418-s19/public/projects/park-chen/local/lib/libopencv_core.so.2.4.9
binarizewolfjolion: CMakeFiles/binarizewolfjolion.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable binarizewolfjolion"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/binarizewolfjolion.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/binarizewolfjolion.dir/build: binarizewolfjolion
.PHONY : CMakeFiles/binarizewolfjolion.dir/build

CMakeFiles/binarizewolfjolion.dir/requires: CMakeFiles/binarizewolfjolion.dir/binarizewolfjolion.cpp.o.requires
CMakeFiles/binarizewolfjolion.dir/requires: CMakeFiles/binarizewolfjolion.dir/timing.cpp.o.requires
.PHONY : CMakeFiles/binarizewolfjolion.dir/requires

CMakeFiles/binarizewolfjolion.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/binarizewolfjolion.dir/cmake_clean.cmake
.PHONY : CMakeFiles/binarizewolfjolion.dir/clean

CMakeFiles/binarizewolfjolion.dir/depend:
	cd /afs/cs.cmu.edu/academic/class/15418-s19/public/projects/park-chen/local_adaptive_binarization/serial && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /afs/cs.cmu.edu/academic/class/15418-s19/public/projects/park-chen/local_adaptive_binarization/serial /afs/cs.cmu.edu/academic/class/15418-s19/public/projects/park-chen/local_adaptive_binarization/serial /afs/cs.cmu.edu/academic/class/15418-s19/public/projects/park-chen/local_adaptive_binarization/serial /afs/cs.cmu.edu/academic/class/15418-s19/public/projects/park-chen/local_adaptive_binarization/serial /afs/cs.cmu.edu/academic/class/15418-s19/public/projects/park-chen/local_adaptive_binarization/serial/CMakeFiles/binarizewolfjolion.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/binarizewolfjolion.dir/depend

