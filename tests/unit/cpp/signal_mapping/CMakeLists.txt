# C++ unit tests for signal_mapping (Catch2).
# Aligned with docs/Tests/Test_New_Plans.md (tests/unit/cpp/signal_mapping/).

cmake_minimum_required(VERSION 3.15)
project(am_qadf_signal_mapping_cpp_tests LANGUAGES CXX)

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

set(SIGNAL_MAPPING_TEST_SOURCES
    test_interpolation_base.cpp
    test_nearest_neighbor.cpp
    test_linear_interpolation.cpp
    test_idw_interpolation.cpp
    test_kde_interpolation.cpp
    test_rbf_interpolation.cpp
)

if(TARGET am_qadf_native_static)
    add_executable(test_signal_mapping_cpp ${SIGNAL_MAPPING_TEST_SOURCES})
    target_link_libraries(test_signal_mapping_cpp PRIVATE Catch2::Catch2WithMain am_qadf_native_static)
else()
    add_executable(test_signal_mapping_cpp ${SIGNAL_MAPPING_TEST_SOURCES})
    target_link_libraries(test_signal_mapping_cpp PRIVATE Catch2::Catch2WithMain)
    target_include_directories(test_signal_mapping_cpp PRIVATE
        ${CMAKE_SOURCE_DIR}/src/am_qadf_native/include
    )
    message(STATUS "test_signal_mapping_cpp: link will fail until am_qadf_native_static is added in root")
endif()

target_compile_features(test_signal_mapping_cpp PRIVATE cxx_std_17)

if(EIGEN3_FOUND OR Eigen3_FOUND)
    if(EIGEN3_INCLUDE_DIR)
        target_include_directories(test_signal_mapping_cpp PRIVATE ${EIGEN3_INCLUDE_DIR})
        target_compile_definitions(test_signal_mapping_cpp PRIVATE EIGEN_AVAILABLE)
    elseif(Eigen3_INCLUDE_DIR)
        target_include_directories(test_signal_mapping_cpp PRIVATE ${Eigen3_INCLUDE_DIR})
        target_compile_definitions(test_signal_mapping_cpp PRIVATE EIGEN_AVAILABLE)
    endif()
endif()

include(CTest)
add_test(NAME test_signal_mapping_cpp COMMAND test_signal_mapping_cpp)
set_tests_properties(test_signal_mapping_cpp PROPERTIES LABELS "cpp;unit;signal_mapping")
