# C++ unit tests for voxelization (Catch2).
# Build when BUILD_TESTS=ON; requires am_qadf_native to be built as a static lib or
# tests linked with the same sources. See parent CMakeLists for integration.
#
# Voxelization C++ tests are 100% C++ (OpenVDB, STL, hatching, grid factory). The
# numpy_converter (NumPy <-> OpenVDB) is a Python bridge only; it is tested from
# Python when code uses am_qadf_native.numpy_to_openvdb / openvdb_to_numpy.

cmake_minimum_required(VERSION 3.15)
project(am_qadf_voxelization_cpp_tests LANGUAGES CXX)

# Catch2: FetchContent or find_package
include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.4.0
)
set(CATCH_INSTALL_DOCS OFF CACHE BOOL "" FORCE)
set(CATCH_INSTALL_EXTRAS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(Catch2)

# Include and link: use parent's native target if available
# This file is intended to be included from tests/unit/cpp/CMakeLists.txt or
# root CMakeLists.txt that defines am_qadf_native or am_qadf_native_static.

set(VOXELIZATION_TEST_SOURCES
    test_openvdb_grid.cpp
    test_unified_grid_factory.cpp
    test_hatching_voxelizer.cpp
    test_stl_voxelizer.cpp
    test_grid_factory.cpp
)

# Prefer static lib from root; otherwise tests will need to be built from root
# with am_qadf_native_static (see tests/unit/cpp/README.md).
if(TARGET am_qadf_native_static)
    add_executable(test_voxelization_cpp ${VOXELIZATION_TEST_SOURCES})
    target_link_libraries(test_voxelization_cpp PRIVATE Catch2::Catch2WithMain am_qadf_native_static)
else()
    add_executable(test_voxelization_cpp ${VOXELIZATION_TEST_SOURCES})
    target_link_libraries(test_voxelization_cpp PRIVATE Catch2::Catch2WithMain)
    target_include_directories(test_voxelization_cpp PRIVATE
        ${CMAKE_SOURCE_DIR}/src/am_qadf_native/include
    )
    message(STATUS "test_voxelization_cpp: link will fail until am_qadf_native_static is added in root (see tests/unit/cpp/README.md)")
endif()

target_compile_features(test_voxelization_cpp PRIVATE cxx_std_17)

# Absolute path to cube.stl so tests find the fixture when run from build/ (any cwd).
set(AM_QADF_STL_FIXTURE "${CMAKE_SOURCE_DIR}/tests/fixtures/stl/cube.stl")
string(REPLACE "\\" "/" AM_QADF_STL_FIXTURE "${AM_QADF_STL_FIXTURE}")
string(REPLACE "\"" "\\\"" AM_QADF_STL_FIXTURE_ESC "${AM_QADF_STL_FIXTURE}")
target_compile_definitions(test_voxelization_cpp PRIVATE "AM_QADF_STL_FIXTURE_PATH=\"${AM_QADF_STL_FIXTURE_ESC}\"")

if(EIGEN3_FOUND OR Eigen3_FOUND)
    if(EIGEN3_INCLUDE_DIR)
        target_include_directories(test_voxelization_cpp PRIVATE ${EIGEN3_INCLUDE_DIR})
        target_compile_definitions(test_voxelization_cpp PRIVATE EIGEN_AVAILABLE)
    elseif(Eigen3_INCLUDE_DIR)
        target_include_directories(test_voxelization_cpp PRIVATE ${Eigen3_INCLUDE_DIR})
        target_compile_definitions(test_voxelization_cpp PRIVATE EIGEN_AVAILABLE)
    endif()
endif()

include(CTest)
add_test(NAME test_voxelization_cpp COMMAND test_voxelization_cpp)
set_tests_properties(test_voxelization_cpp PROPERTIES
    LABELS "cpp;unit;voxelization"
    TIMEOUT 120
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
