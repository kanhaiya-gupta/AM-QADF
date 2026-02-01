# C++ unit tests for fusion (Catch2).
# Aligned with docs/Tests/Test_New_Plans.md (tests/unit/cpp/fusion/).

cmake_minimum_required(VERSION 3.15)
project(am_qadf_fusion_cpp_tests LANGUAGES CXX)

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

set(FUSION_TEST_SOURCES
    test_grid_fusion.cpp
    test_fusion_strategies.cpp
    test_fusion_quality.cpp
)

if(TARGET am_qadf_native_static)
    add_executable(test_fusion_cpp ${FUSION_TEST_SOURCES})
    target_link_libraries(test_fusion_cpp PRIVATE Catch2::Catch2WithMain am_qadf_native_static)
else()
    add_executable(test_fusion_cpp ${FUSION_TEST_SOURCES})
    target_link_libraries(test_fusion_cpp PRIVATE Catch2::Catch2WithMain)
    target_include_directories(test_fusion_cpp PRIVATE
        ${CMAKE_SOURCE_DIR}/src/am_qadf_native/include
    )
    message(STATUS "test_fusion_cpp: link will fail until am_qadf_native_static is added in root")
endif()

target_compile_features(test_fusion_cpp PRIVATE cxx_std_17)

include(CTest)
add_test(NAME test_fusion_cpp COMMAND test_fusion_cpp)
set_tests_properties(test_fusion_cpp PROPERTIES LABELS "cpp;unit;fusion")
