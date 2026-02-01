# C++ unit tests for io (Catch2).
# Aligned with docs/Tests/Test_New_Plans.md (tests/unit/cpp/io/).
# openvdb_reader.hpp requires pybind11 (for extractFeatures etc.); tests use loadGridByName/loadAllGrids only.
# mongodb_writer tests built when EIGEN_AVAILABLE; skip when MongoDB unavailable.

cmake_minimum_required(VERSION 3.15)
project(am_qadf_io_cpp_tests LANGUAGES CXX)

# Catch2
include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.4.0
)
set(CATCH_INSTALL_DOCS OFF CACHE BOOL "" FORCE)
set(CATCH_INSTALL_EXTRAS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(Catch2)

set(IO_TEST_SOURCES
    test_openvdb_reader.cpp
    test_vdb_writer.cpp
    test_paraview_exporter.cpp
    test_mongodb_writer.cpp
)

if(TARGET am_qadf_native_static)
    add_executable(test_io_cpp ${IO_TEST_SOURCES})
    target_link_libraries(test_io_cpp PRIVATE Catch2::Catch2WithMain am_qadf_native_static)
else()
    add_executable(test_io_cpp ${IO_TEST_SOURCES})
    target_link_libraries(test_io_cpp PRIVATE Catch2::Catch2WithMain)
    target_include_directories(test_io_cpp PRIVATE
        ${CMAKE_SOURCE_DIR}/src/am_qadf_native/include
    )
    message(STATUS "test_io_cpp: link will fail until am_qadf_native_static is added in root")
endif()

target_compile_features(test_io_cpp PRIVATE cxx_std_17)

# mongodb_writer tests require EIGEN_AVAILABLE
find_package(Eigen3 QUIET)
if(EIGEN3_FOUND OR Eigen3_FOUND)
    target_compile_definitions(test_io_cpp PRIVATE EIGEN_AVAILABLE)
    if(TARGET Eigen3::Eigen)
        target_link_libraries(test_io_cpp PRIVATE Eigen3::Eigen)
    elseif(EIGEN3_INCLUDE_DIR)
        target_include_directories(test_io_cpp PRIVATE ${EIGEN3_INCLUDE_DIR})
    elseif(Eigen3_INCLUDE_DIR)
        target_include_directories(test_io_cpp PRIVATE ${Eigen3_INCLUDE_DIR})
    endif()
endif()

include(CTest)
add_test(NAME test_io_cpp COMMAND test_io_cpp)
set_tests_properties(test_io_cpp PROPERTIES LABELS "cpp;unit;io")
