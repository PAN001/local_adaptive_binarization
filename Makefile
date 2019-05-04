# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

# Default target executed when no arguments are given to make.
default_target: all
.PHONY : default_target

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
CMAKE_SOURCE_DIR = /afs/cs.cmu.edu/academic/class/15418-s19/public/projects/park-chen/local_adaptive_binarization

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /afs/cs.cmu.edu/academic/class/15418-s19/public/projects/park-chen/local_adaptive_binarization

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake cache editor..."
	/usr/bin/ccmake -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache
.PHONY : edit_cache/fast

# Special rule for the target package
package: preinstall
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Run CPack packaging tool..."
	/usr/bin/cpack --config ./CPackConfig.cmake
.PHONY : package

# Special rule for the target package
package/fast: package
.PHONY : package/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/bin/cmake -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache
.PHONY : rebuild_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /afs/cs.cmu.edu/academic/class/15418-s19/public/projects/park-chen/local_adaptive_binarization/CMakeFiles /afs/cs.cmu.edu/academic/class/15418-s19/public/projects/park-chen/local_adaptive_binarization/CMakeFiles/progress.marks
	$(MAKE) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /afs/cs.cmu.edu/academic/class/15418-s19/public/projects/park-chen/local_adaptive_binarization/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean
.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named binarizewolfjolion

# Build rule for target.
binarizewolfjolion: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 binarizewolfjolion
.PHONY : binarizewolfjolion

# fast build rule for target.
binarizewolfjolion/fast:
	$(MAKE) -f CMakeFiles/binarizewolfjolion.dir/build.make CMakeFiles/binarizewolfjolion.dir/build
.PHONY : binarizewolfjolion/fast

#=============================================================================
# Target rules for targets named binarizewolfjolion_sauvola

# Build rule for target.
binarizewolfjolion_sauvola: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 binarizewolfjolion_sauvola
.PHONY : binarizewolfjolion_sauvola

# fast build rule for target.
binarizewolfjolion_sauvola/fast:
	$(MAKE) -f CMakeFiles/binarizewolfjolion_sauvola.dir/build.make CMakeFiles/binarizewolfjolion_sauvola.dir/build
.PHONY : binarizewolfjolion_sauvola/fast

main.o: main.cpp.o
.PHONY : main.o

# target to build an object file
main.cpp.o:
	$(MAKE) -f CMakeFiles/binarizewolfjolion.dir/build.make CMakeFiles/binarizewolfjolion.dir/main.cpp.o
	$(MAKE) -f CMakeFiles/binarizewolfjolion_sauvola.dir/build.make CMakeFiles/binarizewolfjolion_sauvola.dir/main.cpp.o
.PHONY : main.cpp.o

main.i: main.cpp.i
.PHONY : main.i

# target to preprocess a source file
main.cpp.i:
	$(MAKE) -f CMakeFiles/binarizewolfjolion.dir/build.make CMakeFiles/binarizewolfjolion.dir/main.cpp.i
	$(MAKE) -f CMakeFiles/binarizewolfjolion_sauvola.dir/build.make CMakeFiles/binarizewolfjolion_sauvola.dir/main.cpp.i
.PHONY : main.cpp.i

main.s: main.cpp.s
.PHONY : main.s

# target to generate assembly for a file
main.cpp.s:
	$(MAKE) -f CMakeFiles/binarizewolfjolion.dir/build.make CMakeFiles/binarizewolfjolion.dir/main.cpp.s
	$(MAKE) -f CMakeFiles/binarizewolfjolion_sauvola.dir/build.make CMakeFiles/binarizewolfjolion_sauvola.dir/main.cpp.s
.PHONY : main.cpp.s

timing.o: timing.cpp.o
.PHONY : timing.o

# target to build an object file
timing.cpp.o:
	$(MAKE) -f CMakeFiles/binarizewolfjolion.dir/build.make CMakeFiles/binarizewolfjolion.dir/timing.cpp.o
	$(MAKE) -f CMakeFiles/binarizewolfjolion_sauvola.dir/build.make CMakeFiles/binarizewolfjolion_sauvola.dir/timing.cpp.o
.PHONY : timing.cpp.o

timing.i: timing.cpp.i
.PHONY : timing.i

# target to preprocess a source file
timing.cpp.i:
	$(MAKE) -f CMakeFiles/binarizewolfjolion.dir/build.make CMakeFiles/binarizewolfjolion.dir/timing.cpp.i
	$(MAKE) -f CMakeFiles/binarizewolfjolion_sauvola.dir/build.make CMakeFiles/binarizewolfjolion_sauvola.dir/timing.cpp.i
.PHONY : timing.cpp.i

timing.s: timing.cpp.s
.PHONY : timing.s

# target to generate assembly for a file
timing.cpp.s:
	$(MAKE) -f CMakeFiles/binarizewolfjolion.dir/build.make CMakeFiles/binarizewolfjolion.dir/timing.cpp.s
	$(MAKE) -f CMakeFiles/binarizewolfjolion_sauvola.dir/build.make CMakeFiles/binarizewolfjolion_sauvola.dir/timing.cpp.s
.PHONY : timing.cpp.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... binarizewolfjolion"
	@echo "... binarizewolfjolion_sauvola"
	@echo "... edit_cache"
	@echo "... package"
	@echo "... rebuild_cache"
	@echo "... main.o"
	@echo "... main.i"
	@echo "... main.s"
	@echo "... timing.o"
	@echo "... timing.i"
	@echo "... timing.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

