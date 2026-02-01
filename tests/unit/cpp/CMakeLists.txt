# C++ unit tests (Catch2).
# Include from root CMakeLists.txt when BUILD_TESTS=ON and C++ tests are enabled.
# Requires: Catch2 (FetchContent in subdirs), and either am_qadf_native_static
# (add a static lib in root from same sources as am_qadf_native) or duplicate
# source listing in each test subdir.

cmake_minimum_required(VERSION 3.15)
project(am_qadf_cpp_tests LANGUAGES CXX)

enable_testing()
add_subdirectory(voxelization)
add_subdirectory(signal_mapping)
add_subdirectory(fusion)
add_subdirectory(synchronization)
add_subdirectory(query)
add_subdirectory(correction)
add_subdirectory(processing)
add_subdirectory(io)