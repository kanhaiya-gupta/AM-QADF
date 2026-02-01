# C++ performance benchmarks (Google Benchmark).
# Build when ENABLE_BENCHMARKS=ON and am_qadf_native_static is defined.
# Aligned with docs/Tests/Test_New_Plans.md (tests/performance/cpp/).

cmake_minimum_required(VERSION 3.15)
project(am_qadf_performance_cpp LANGUAGES CXX)

# Use existing benchmark target when provided by parent (FetchContent or find_package)
if(NOT TARGET benchmark::benchmark)
    find_package(benchmark REQUIRED)
endif()

set(BENCHMARK_SOURCES
    benchmark_voxelization.cpp
    benchmark_signal_mapping.cpp
    benchmark_fusion.cpp
    benchmark_synchronization.cpp
    benchmark_query.cpp
)

foreach(SRC ${BENCHMARK_SOURCES})
    get_filename_component(NAME ${SRC} NAME_WE)
    string(REPLACE "benchmark_" "" EXE_NAME "${NAME}")
    if(TARGET am_qadf_native_static)
        add_executable(bench_${EXE_NAME} ${SRC})
        target_link_libraries(bench_${EXE_NAME} PRIVATE
            benchmark::benchmark
            benchmark::benchmark_main
            am_qadf_native_static
        )
    else()
        add_executable(bench_${EXE_NAME} ${SRC})
        target_link_libraries(bench_${EXE_NAME} PRIVATE
            benchmark::benchmark
            benchmark::benchmark_main
        )
        target_include_directories(bench_${EXE_NAME} PRIVATE
            ${CMAKE_SOURCE_DIR}/src/am_qadf_native/include
        )
        message(STATUS "bench_${EXE_NAME}: link will fail until am_qadf_native_static is added in root")
    endif()
    target_compile_features(bench_${EXE_NAME} PRIVATE cxx_std_17)
    # Register with CTest so "ctest --test-dir build" can run benchmarks (label: benchmark)
    add_test(NAME bench_${EXE_NAME} COMMAND bench_${EXE_NAME} --benchmark_min_time=0.01)
    set_tests_properties(bench_${EXE_NAME} PROPERTIES LABELS "benchmark;performance")
endforeach()

# Eigen (synchronization benchmarks)
find_package(Eigen3 QUIET)
if(EIGEN3_FOUND OR Eigen3_FOUND)
    foreach(SRC ${BENCHMARK_SOURCES})
        get_filename_component(NAME ${SRC} NAME_WE)
        string(REPLACE "benchmark_" "" EXE_NAME "${NAME}")
        if(TARGET bench_${EXE_NAME})
            target_compile_definitions(bench_${EXE_NAME} PRIVATE EIGEN_AVAILABLE)
            if(TARGET Eigen3::Eigen)
                target_link_libraries(bench_${EXE_NAME} PRIVATE Eigen3::Eigen)
            elseif(EIGEN3_INCLUDE_DIR)
                target_include_directories(bench_${EXE_NAME} PRIVATE ${EIGEN3_INCLUDE_DIR})
            elseif(Eigen3_INCLUDE_DIR)
                target_include_directories(bench_${EXE_NAME} PRIVATE ${Eigen3_INCLUDE_DIR})
            endif()
        endif()
    endforeach()
endif()
