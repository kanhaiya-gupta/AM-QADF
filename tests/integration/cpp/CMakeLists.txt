# C++ integration tests (Catch2).
# Chains multiple native modules: e.g. voxelization → io, signal_mapping → io, fusion → io.
# Include from root when BUILD_TESTS=ON and am_qadf_native_static is defined (e.g. add_subdirectory(tests/integration/cpp)).
# Aligned with docs/Tests/Test_New_Plans.md (tests/integration/cpp/).

cmake_minimum_required(VERSION 3.15)
project(am_qadf_integration_cpp_tests LANGUAGES CXX)

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

set(INTEGRATION_TEST_SOURCES
    test_openvdb_integration.cpp
    test_grid_operations.cpp
    test_signal_mapping_pipeline.cpp
    test_fusion_pipeline.cpp
    test_synchronization_pipeline.cpp
    test_correction_pipeline.cpp
    test_processing_pipeline.cpp
)

if(TARGET am_qadf_native_static)
    add_executable(test_integration_cpp ${INTEGRATION_TEST_SOURCES})
    target_link_libraries(test_integration_cpp PRIVATE Catch2::Catch2WithMain am_qadf_native_static)
else()
    add_executable(test_integration_cpp ${INTEGRATION_TEST_SOURCES})
    target_link_libraries(test_integration_cpp PRIVATE Catch2::Catch2WithMain)
    target_include_directories(test_integration_cpp PRIVATE
        ${CMAKE_SOURCE_DIR}/src/am_qadf_native/include
    )
    message(STATUS "test_integration_cpp: link will fail until am_qadf_native_static is added in root")
endif()

target_compile_features(test_integration_cpp PRIVATE cxx_std_17)

# Eigen (synchronization, optional)
find_package(Eigen3 QUIET)
if(EIGEN3_FOUND OR Eigen3_FOUND)
    target_compile_definitions(test_integration_cpp PRIVATE EIGEN_AVAILABLE)
    if(TARGET Eigen3::Eigen)
        target_link_libraries(test_integration_cpp PRIVATE Eigen3::Eigen)
    elseif(EIGEN3_INCLUDE_DIR)
        target_include_directories(test_integration_cpp PRIVATE ${EIGEN3_INCLUDE_DIR})
    elseif(Eigen3_INCLUDE_DIR)
        target_include_directories(test_integration_cpp PRIVATE ${Eigen3_INCLUDE_DIR})
    endif()
endif()

include(CTest)
add_test(NAME test_integration_cpp COMMAND test_integration_cpp)
set_tests_properties(test_integration_cpp PROPERTIES LABELS "cpp;integration")
