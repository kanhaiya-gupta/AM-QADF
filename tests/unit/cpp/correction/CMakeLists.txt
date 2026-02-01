# C++ unit tests for correction (Catch2).
# Aligned with docs/Tests/Test_New_Plans.md (tests/unit/cpp/correction/).

cmake_minimum_required(VERSION 3.15)
project(am_qadf_correction_cpp_tests LANGUAGES CXX)

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

set(CORRECTION_TEST_SOURCES
    test_signal_noise_reduction.cpp
    test_spatial_noise_filtering.cpp
    test_geometric_correction.cpp
    test_calibration.cpp
    test_validation.cpp
)

if(TARGET am_qadf_native_static)
    add_executable(test_correction_cpp ${CORRECTION_TEST_SOURCES})
    target_link_libraries(test_correction_cpp PRIVATE Catch2::Catch2WithMain am_qadf_native_static)
else()
    add_executable(test_correction_cpp ${CORRECTION_TEST_SOURCES})
    target_link_libraries(test_correction_cpp PRIVATE Catch2::Catch2WithMain)
    target_include_directories(test_correction_cpp PRIVATE
        ${CMAKE_SOURCE_DIR}/src/am_qadf_native/include
    )
    message(STATUS "test_correction_cpp: link will fail until am_qadf_native_static is added in root")
endif()

target_compile_features(test_correction_cpp PRIVATE cxx_std_17)

include(CTest)
add_test(NAME test_correction_cpp COMMAND test_correction_cpp)
set_tests_properties(test_correction_cpp PROPERTIES LABELS "cpp;unit;correction")
