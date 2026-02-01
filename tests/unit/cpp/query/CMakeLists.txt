# C++ unit tests for query (Catch2).
# Aligned with docs/Tests/Test_New_Plans.md (tests/unit/cpp/query/).
# MongoDB tests skip when server unavailable. point_converter built when EIGEN_AVAILABLE.

cmake_minimum_required(VERSION 3.15)
project(am_qadf_query_cpp_tests LANGUAGES CXX)

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

set(QUERY_TEST_SOURCES
    test_query_result.cpp
    test_point_converter.cpp
    test_ct_image_reader.cpp
    test_mongodb_query_client.cpp
    test_ct_scan_query.cpp
    test_laser_monitoring_query.cpp
)

if(TARGET am_qadf_native_static)
    add_executable(test_query_cpp ${QUERY_TEST_SOURCES})
    target_link_libraries(test_query_cpp PRIVATE Catch2::Catch2WithMain am_qadf_native_static)
else()
    add_executable(test_query_cpp ${QUERY_TEST_SOURCES})
    target_link_libraries(test_query_cpp PRIVATE Catch2::Catch2WithMain)
    target_include_directories(test_query_cpp PRIVATE
        ${CMAKE_SOURCE_DIR}/src/am_qadf_native/include
    )
    message(STATUS "test_query_cpp: link will fail until am_qadf_native_static is added in root")
endif()

target_compile_features(test_query_cpp PRIVATE cxx_std_17)

# Use same Eigen as am_qadf_native_static (parent sets EIGEN3_INCLUDE_DIR / EIGEN3_FOUND)
if(EIGEN3_FOUND)
    target_compile_definitions(test_query_cpp PRIVATE EIGEN_AVAILABLE)
    target_include_directories(test_query_cpp PRIVATE ${EIGEN3_INCLUDE_DIR})
else()
    find_package(Eigen3 QUIET)
    if(EIGEN3_FOUND OR Eigen3_FOUND)
        target_compile_definitions(test_query_cpp PRIVATE EIGEN_AVAILABLE)
        if(TARGET Eigen3::Eigen)
            target_link_libraries(test_query_cpp PRIVATE Eigen3::Eigen)
        elseif(EIGEN3_INCLUDE_DIR)
            target_include_directories(test_query_cpp PRIVATE ${EIGEN3_INCLUDE_DIR})
        elseif(Eigen3_INCLUDE_DIR)
            target_include_directories(test_query_cpp PRIVATE ${Eigen3_INCLUDE_DIR})
        endif()
    endif()
endif()

include(CTest)
add_test(NAME test_query_cpp COMMAND test_query_cpp)
set_tests_properties(test_query_cpp PROPERTIES LABELS "cpp;unit;query")
