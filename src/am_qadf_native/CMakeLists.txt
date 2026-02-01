# Source files (voxelization is 100% C++; NumPy bridge is in BRIDGE_SOURCES, Python module only)
set(VOXELIZATION_SOURCES
    src/voxelization/openvdb_grid.cpp
    src/voxelization/stl_voxelizer.cpp
    src/voxelization/hatching_voxelizer.cpp
    src/voxelization/unified_grid_factory.cpp
)

# Python-only bridge: NumPy <-> OpenVDB (not in static lib; avoids Python in C++ voxelization tests)
set(BRIDGE_SOURCES
    src/bridge/numpy_openvdb_bridge.cpp
)

set(SIGNAL_MAPPING_SOURCES
    src/signal_mapping/interpolation_base.cpp
    src/signal_mapping/nearest_neighbor.cpp
    src/signal_mapping/linear_interpolation.cpp
    src/signal_mapping/idw_interpolation.cpp
    src/signal_mapping/kde_interpolation.cpp
    src/signal_mapping/rbf_interpolation.cpp
)

set(FUSION_SOURCES
    src/fusion/grid_fusion.cpp
    src/fusion/fusion_strategies.cpp
    src/fusion/fusion_quality.cpp
)

set(SYNCHRONIZATION_SOURCES
    src/synchronization/grid_spatial_alignment.cpp
    src/synchronization/grid_temporal_alignment.cpp
    src/synchronization/grid_synchronizer.cpp
    src/synchronization/point_transformation_estimate.cpp
    src/synchronization/point_transformation_validate.cpp
    src/synchronization/point_transformation_sampling.cpp
    src/synchronization/point_transform.cpp
    src/synchronization/point_bounds.cpp
    src/synchronization/point_coordinate_transform.cpp
    src/synchronization/point_temporal_alignment.cpp
)

set(QUERY_SOURCES
    src/query/mongodb_query_client.cpp
    src/query/ct_image_reader.cpp
    src/query/laser_monitoring_query.cpp
    src/query/ct_scan_query.cpp
    src/query/point_converter.cpp
)

set(VISUALIZATION_SOURCES
    src/visualization/hatching_visualization_data.cpp
    src/visualization/point_cloud_visualization_data.cpp
)

set(CORRECTION_SOURCES
    src/correction/signal_noise_reduction.cpp
    src/correction/spatial_noise_filtering.cpp
    src/correction/geometric_correction.cpp
    src/correction/calibration.cpp
    src/correction/validation.cpp
)

set(PROCESSING_SOURCES
    src/processing/signal_processing.cpp
    src/processing/signal_generation.cpp
)

set(IO_SOURCES
    src/io/mongocxx_instance.cpp
    src/io/save_processed_points_bridge.cpp
    src/io/openvdb_reader.cpp
    src/io/vdb_writer.cpp
    src/io/paraview_exporter.cpp
    src/io/mongodb_writer.cpp
)

# Static library for C++ tests (same C++ sources, no Python bindings).
# Built when BUILD_TESTS=ON; tests link to am_qadf_native_static.
add_library(am_qadf_native_static STATIC
    ${VOXELIZATION_SOURCES}
    ${SIGNAL_MAPPING_SOURCES}
    ${FUSION_SOURCES}
    ${SYNCHRONIZATION_SOURCES}
    ${QUERY_SOURCES}
    ${VISUALIZATION_SOURCES}
    ${CORRECTION_SOURCES}
    ${PROCESSING_SOURCES}
    ${IO_SOURCES}
)
set_target_properties(am_qadf_native_static PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(am_qadf_native_static PRIVATE cxx_std_17)
# PUBLIC so test executables that link to this get the include path for am_qadf_native/*.hpp
target_include_directories(am_qadf_native_static PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
# Same sources (and test .cpp that include our headers) need Python/pybind11 and mongocxx/ITK includes
target_link_libraries(am_qadf_native_static PUBLIC Python3::Python pybind11::headers)

# Python bindings
set(PYTHON_BINDINGS
    python/bindings/voxelization_bindings.cpp
    python/bindings/signal_mapping_bindings.cpp
    python/bindings/fusion_bindings.cpp
    python/bindings/synchronization_bindings.cpp
    python/bindings/query_bindings.cpp
    python/bindings/visualization_bindings.cpp
    python/bindings/correction_bindings.cpp
    python/bindings/processing_bindings.cpp
    python/bindings/io_bindings.cpp
    python/module.cpp
)

# Create Python module (includes BRIDGE_SOURCES for numpy_to_openvdb / openvdb_to_numpy)
pybind11_add_module(am_qadf_native
    ${VOXELIZATION_SOURCES}
    ${BRIDGE_SOURCES}
    ${SIGNAL_MAPPING_SOURCES}
    ${FUSION_SOURCES}
    ${SYNCHRONIZATION_SOURCES}
    ${QUERY_SOURCES}
    ${VISUALIZATION_SOURCES}
    ${CORRECTION_SOURCES}
    ${PROCESSING_SOURCES}
    ${IO_SOURCES}
    ${PYTHON_BINDINGS}
)

# Modern CMake target-based approach
target_compile_features(am_qadf_native PRIVATE cxx_std_17)

# Include directories
target_include_directories(am_qadf_native PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Add Eigen include directory if found
if(EIGEN3_FOUND OR Eigen3_FOUND)
    if(EIGEN3_INCLUDE_DIR)
        target_include_directories(am_qadf_native PRIVATE ${EIGEN3_INCLUDE_DIR})
        target_compile_definitions(am_qadf_native PRIVATE EIGEN_AVAILABLE)
        target_include_directories(am_qadf_native_static PRIVATE ${EIGEN3_INCLUDE_DIR})
        target_compile_definitions(am_qadf_native_static PRIVATE EIGEN_AVAILABLE)
        message(STATUS "Eigen include directory: ${EIGEN3_INCLUDE_DIR}")
    elseif(Eigen3_INCLUDE_DIR)
        target_include_directories(am_qadf_native PRIVATE ${Eigen3_INCLUDE_DIR})
        target_compile_definitions(am_qadf_native PRIVATE EIGEN_AVAILABLE)
        target_include_directories(am_qadf_native_static PRIVATE ${Eigen3_INCLUDE_DIR})
        target_compile_definitions(am_qadf_native_static PRIVATE EIGEN_AVAILABLE)
        message(STATUS "Eigen include directory: ${Eigen3_INCLUDE_DIR}")
    endif()
endif()

# Add KFR include directory if found
if(KFR_FOUND)
    target_include_directories(am_qadf_native PRIVATE ${KFR_INCLUDE_DIR})
    target_compile_definitions(am_qadf_native PRIVATE KFR_AVAILABLE)
    target_include_directories(am_qadf_native_static PRIVATE ${KFR_INCLUDE_DIR})
    target_compile_definitions(am_qadf_native_static PRIVATE KFR_AVAILABLE)
    message(STATUS "KFR include directory: ${KFR_INCLUDE_DIR}")
endif()

# Add TBB FIRST (required by OpenVDB headers)
# CRITICAL: TBB must be added BEFORE OpenVDB so OpenVDB headers can find TBB
# TBB_INCLUDE_DIR should be set in parent CMakeLists.txt and cached
if(TBB_FOUND OR TBB_INCLUDE_DIR)
    if(TARGET TBB::tbb)
        target_link_libraries(am_qadf_native PRIVATE TBB::tbb)
        target_link_libraries(am_qadf_native_static PRIVATE TBB::tbb)
        message(STATUS "TBB linked via imported target")
    else()
        # Fallback: add include directory and link library
        # CRITICAL: Must add include directory BEFORE OpenVDB for headers to find TBB
        # Read TBB_INCLUDE_DIR from CACHE (set by parent CMakeLists.txt)
        get_property(TBB_INCLUDE_DIR_CACHED CACHE TBB_INCLUDE_DIR PROPERTY VALUE)
        if(TBB_INCLUDE_DIR_CACHED)
            set(TBB_INCLUDE_DIR ${TBB_INCLUDE_DIR_CACHED})
        elseif(NOT TBB_INCLUDE_DIR)
            # Re-find if not in cache
            find_path(TBB_INCLUDE_DIR tbb/blocked_range.h
                PATHS
                $ENV{CONDA_PREFIX}/include
                /usr/include
                /usr/local/include
                /opt/intel/oneapi/tbb/latest/include
            )
        endif()
        
        # Verify the header file actually exists and add include directory
        if(TBB_INCLUDE_DIR AND EXISTS "${TBB_INCLUDE_DIR}/tbb/blocked_range.h")
            # If TBB_NEEDS_ISYSTEM is set (conda compiler with system TBB), use -isystem
            if(TBB_NEEDS_ISYSTEM)
                # Use -isystem for conda compiler to access system TBB
                target_include_directories(am_qadf_native SYSTEM PRIVATE ${TBB_INCLUDE_DIR})
                if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
                    target_compile_options(am_qadf_native PRIVATE -isystem ${TBB_INCLUDE_DIR})
                endif()
                message(STATUS "TBB include directory added (SYSTEM, -isystem flag): ${TBB_INCLUDE_DIR}")
            else()
                # CRITICAL: Add TBB includes as REGULAR include directories so they're searched BEFORE OpenVDB headers
                # Regular includes are searched before SYSTEM includes, ensuring OpenVDB finds TBB
                # Add as regular include directory AND via compile options to ensure it's definitely found
                target_include_directories(am_qadf_native PRIVATE ${TBB_INCLUDE_DIR})
                # Always add via compile options to ensure it's in the search path
                if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
                    target_compile_options(am_qadf_native PRIVATE -I${TBB_INCLUDE_DIR})
                elseif(MSVC)
                    target_compile_options(am_qadf_native PRIVATE /I${TBB_INCLUDE_DIR})
                endif()
                if(TBB_NEEDS_EXPLICIT_INCLUDE)
                    message(STATUS "TBB include directory added (PRIVATE + compile option, explicit flag needed): ${TBB_INCLUDE_DIR}")
                else()
                    message(STATUS "TBB include directory added (PRIVATE + compile option): ${TBB_INCLUDE_DIR}")
                endif()
            endif()
            message(STATUS "  Verified: ${TBB_INCLUDE_DIR}/tbb/blocked_range.h exists")
        elseif(TBB_INCLUDE_DIR AND EXISTS "${TBB_INCLUDE_DIR}/oneapi/tbb/blocked_range.h")
            # Intel oneAPI TBB path
            target_include_directories(am_qadf_native PRIVATE ${TBB_INCLUDE_DIR})
            if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
                target_compile_options(am_qadf_native PRIVATE -I${TBB_INCLUDE_DIR})
            elseif(MSVC)
                target_compile_options(am_qadf_native PRIVATE /I${TBB_INCLUDE_DIR})
            endif()
            if(TBB_NEEDS_EXPLICIT_INCLUDE)
                message(STATUS "TBB include directory added (PRIVATE + compile option, oneAPI, explicit flag needed): ${TBB_INCLUDE_DIR}")
            else()
                message(STATUS "TBB include directory added (PRIVATE + compile option, oneAPI): ${TBB_INCLUDE_DIR}")
            endif()
            message(STATUS "  Verified: ${TBB_INCLUDE_DIR}/oneapi/tbb/blocked_range.h exists")
            target_compile_definitions(am_qadf_native PRIVATE TBB_USE_ONEAPI)
        else()
            message(FATAL_ERROR "TBB include directory not found or invalid!")
            message(FATAL_ERROR "  TBB_INCLUDE_DIR=${TBB_INCLUDE_DIR}")
            if(TBB_INCLUDE_DIR)
                message(FATAL_ERROR "  Checked: ${TBB_INCLUDE_DIR}/tbb/blocked_range.h")
                message(FATAL_ERROR "  Checked: ${TBB_INCLUDE_DIR}/oneapi/tbb/blocked_range.h")
            endif()
            message(FATAL_ERROR "  Install via: sudo apt-get install libtbb-dev")
            message(FATAL_ERROR "  Or: conda install -c conda-forge tbb")
        endif()
        
        # Try to find and link TBB library
        # Search in common system library paths
        if(NOT DEFINED TBB_LIB OR NOT TBB_LIB)
            find_library(TBB_LIB tbb 
                PATHS 
                $ENV{CONDA_PREFIX}/lib 
                /usr/lib/x86_64-linux-gnu
                /usr/lib
                /usr/local/lib
                /opt/intel/oneapi/tbb/latest/lib
                NO_DEFAULT_PATH
            )
        endif()
        if(TBB_LIB)
            target_link_libraries(am_qadf_native PRIVATE ${TBB_LIB})
            message(STATUS "TBB linked: ${TBB_LIB}")
        else()
            # Try without NO_DEFAULT_PATH as last resort
            find_library(TBB_LIB tbb)
            if(TBB_LIB)
                target_link_libraries(am_qadf_native PRIVATE ${TBB_LIB})
                message(STATUS "TBB linked: ${TBB_LIB}")
            else()
                message(WARNING "TBB library not found, but headers found - linking may fail")
            endif()
        endif()
        # Same TBB for static lib (C++ tests)
        if(TBB_INCLUDE_DIR)
            target_include_directories(am_qadf_native_static PRIVATE ${TBB_INCLUDE_DIR})
            if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
                target_compile_options(am_qadf_native_static PRIVATE -I${TBB_INCLUDE_DIR})
            elseif(MSVC)
                target_compile_options(am_qadf_native_static PRIVATE /I${TBB_INCLUDE_DIR})
            endif()
        endif()
        if(TBB_LIB)
            target_link_libraries(am_qadf_native_static PRIVATE ${TBB_LIB})
        endif()
    endif()
    target_compile_definitions(am_qadf_native PRIVATE TBB_AVAILABLE)
    target_compile_definitions(am_qadf_native_static PRIVATE TBB_AVAILABLE)
else()
    message(FATAL_ERROR "TBB is required by OpenVDB but not found. Install via: conda install -c conda-forge tbb")
endif()

# Link libraries with modern CMake
# NOTE: TBB must be added BEFORE OpenVDB (done above)
target_link_libraries(am_qadf_native PRIVATE
    Threads::Threads
    Python3::Python
    pybind11::module
    OpenVDB::openvdb
)
# PUBLIC so test executables get OpenVDB/Threads includes and link (they compile headers that include openvdb/openvdb.h)
target_link_libraries(am_qadf_native_static PUBLIC Threads::Threads OpenVDB::openvdb)

# Link ITK (REQUIRED); PUBLIC on static so tests that include our headers get ITK
if(ITK_FOUND)
    target_include_directories(am_qadf_native PRIVATE ${ITK_INCLUDE_DIRS})
    target_link_libraries(am_qadf_native PRIVATE ${ITK_LIBRARIES})
    target_include_directories(am_qadf_native_static PUBLIC ${ITK_INCLUDE_DIRS})
    target_link_libraries(am_qadf_native_static PUBLIC ${ITK_LIBRARIES})
    target_compile_definitions(am_qadf_native PRIVATE ITK_AVAILABLE=1)
    target_compile_definitions(am_qadf_native_static PRIVATE ITK_AVAILABLE=1)
    message(STATUS "ITK found - CT scan reading enabled")
    message(STATUS "  ITK include directories: ${ITK_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "ITK is REQUIRED but not found! Build will fail.")
    message(STATUS "  Install ITK:")
    message(STATUS "    - Build from source: See docs/Infrastructure/INSTALL_DEPENDENCIES.md")
    message(STATUS "    - Or: conda install -c conda-forge itk")
    message(STATUS "  Set ITK_DIR when running cmake:")
    message(STATUS "    cmake -DITK_DIR=/path/to/ITK/build ..")
endif()

# Link mongocxx (REQUIRED)
if(mongocxx_FOUND)
    # Try to use imported targets first (preferred)
    # Check if targets exist (they should be created by find_package)
    if(TARGET mongocxx::mongocxx AND TARGET bsoncxx::bsoncxx)
        target_link_libraries(am_qadf_native PRIVATE
            mongocxx::mongocxx
            bsoncxx::bsoncxx
        )
        target_link_libraries(am_qadf_native_static PUBLIC mongocxx::mongocxx bsoncxx::bsoncxx)
        message(STATUS "mongocxx linked via imported targets")
    else()
        # Fallback: link libraries directly and set up include directories manually
        # Get install directory from parent CMakeLists.txt or use default
        if(NOT DEFINED MONGOCXX_INSTALL_DIR)
            set(MONGOCXX_INSTALL_DIR "/mnt/c/Users/kanha/Independent_Research/AM-QADF/third_party/mongo-cxx-driver/install")
        endif()
        
        find_library(MONGOCXX_LIB mongocxx PATHS "${MONGOCXX_INSTALL_DIR}/lib" NO_DEFAULT_PATH)
        find_library(BSONCXX_LIB bsoncxx PATHS "${MONGOCXX_INSTALL_DIR}/lib" NO_DEFAULT_PATH)
        
        if(MONGOCXX_LIB AND BSONCXX_LIB)
            target_link_libraries(am_qadf_native PRIVATE ${MONGOCXX_LIB} ${BSONCXX_LIB})
            target_link_libraries(am_qadf_native_static PUBLIC ${MONGOCXX_LIB} ${BSONCXX_LIB})
            # Add include directories
            # Headers structure: 
            #   - include/mongocxx/v_noabi/mongocxx/client.hpp -> <mongocxx/client.hpp>
            #   - include/bsoncxx/v_noabi/bsoncxx/... -> <bsoncxx/...>
            #   - include/bson-2.1.2/bson/bson.h -> <bson/bson.h>
            #   - include/mongoc-2.1.2/mongoc/... -> <mongoc/...>
            set(MONGOCXX_INCLUDES
                "${MONGOCXX_INSTALL_DIR}/include"
                "${MONGOCXX_INSTALL_DIR}/include/mongocxx/v_noabi"
                "${MONGOCXX_INSTALL_DIR}/include/bsoncxx/v_noabi"
                "${MONGOCXX_INSTALL_DIR}/include/bson-2.1.2"
                "${MONGOCXX_INSTALL_DIR}/include/mongoc-2.1.2"
            )
            target_include_directories(am_qadf_native PRIVATE ${MONGOCXX_INCLUDES})
            target_include_directories(am_qadf_native_static PUBLIC ${MONGOCXX_INCLUDES})
            message(STATUS "mongocxx linked via direct libraries: ${MONGOCXX_LIB}, ${BSONCXX_LIB}")
            message(STATUS "mongocxx include directories:")
            message(STATUS "  - ${MONGOCXX_INSTALL_DIR}/include")
            message(STATUS "  - ${MONGOCXX_INSTALL_DIR}/include/mongocxx/v_noabi")
            message(STATUS "  - ${MONGOCXX_INSTALL_DIR}/include/bsoncxx/v_noabi")
        else()
            message(FATAL_ERROR "mongocxx libraries not found. Checked: ${MONGOCXX_INSTALL_DIR}/lib")
        endif()
    endif()
    target_compile_definitions(am_qadf_native PRIVATE MONGOCXX_AVAILABLE)
    target_compile_definitions(am_qadf_native_static PRIVATE MONGOCXX_AVAILABLE)
else()
    message(FATAL_ERROR "mongocxx is required but not found")
endif()

# mongoc (used by mongocxx) needs Cyrus SASL at link time; add explicitly so linker finds libsasl2.so
if(UNIX AND NOT APPLE AND mongocxx_FOUND)
    find_library(SASL2_LIB NAMES sasl2)
    if(NOT SASL2_LIB)
        find_library(SASL2_LIB NAMES sasl2 PATHS /usr/lib/x86_64-linux-gnu NO_DEFAULT_PATH)
    endif()
    if(SASL2_LIB)
        target_link_libraries(am_qadf_native_static PUBLIC ${SASL2_LIB})
        target_link_libraries(am_qadf_native PRIVATE ${SASL2_LIB})
        message(STATUS "Cyrus SASL linked for mongoc: ${SASL2_LIB}")
    else()
        message(WARNING "libsasl2 not found; link of query/mongo tests may fail. Install libsasl2-dev (apt) or set CMAKE_LIBRARY_PATH")
    endif()
endif()

# Add OpenMP if available
if(MSVC AND OPENMP_AVAILABLE)
    target_compile_options(am_qadf_native PRIVATE /openmp)
    target_compile_definitions(am_qadf_native PRIVATE OPENMP_AVAILABLE)
    target_compile_options(am_qadf_native_static PRIVATE /openmp)
    target_compile_definitions(am_qadf_native_static PRIVATE OPENMP_AVAILABLE)
    message(STATUS "MSVC OpenMP enabled")
elseif(OpenMP_CXX_FOUND)
    target_compile_options(am_qadf_native PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries(am_qadf_native PRIVATE ${OpenMP_CXX_LIBRARIES})
    target_compile_definitions(am_qadf_native PRIVATE OPENMP_AVAILABLE)
    target_compile_options(am_qadf_native_static PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries(am_qadf_native_static PRIVATE ${OpenMP_CXX_LIBRARIES})
    target_compile_definitions(am_qadf_native_static PRIVATE OPENMP_AVAILABLE)
    message(STATUS "OpenMP linked successfully")
else()
    target_compile_definitions(am_qadf_native PRIVATE OPENMP_UNAVAILABLE)
    target_compile_definitions(am_qadf_native_static PRIVATE OPENMP_UNAVAILABLE)
endif()

# Set target properties
set_target_properties(am_qadf_native PROPERTIES
    CXX_VISIBILITY_PRESET "hidden"
    VISIBILITY_INLINES_HIDDEN ON
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Set output directory - keep .pyd/.so files in Release/Debug folders
set_target_properties(am_qadf_native PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_BINARY_DIR}/Debug
    LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_BINARY_DIR}/Release
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_BINARY_DIR}/Debug
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_BINARY_DIR}/Release
)

# Add benchmarking if enabled
if(ENABLE_BENCHMARKS)
    find_package(benchmark QUIET)
    if(benchmark_FOUND)
        target_link_libraries(am_qadf_native PRIVATE benchmark::benchmark)
        target_compile_definitions(am_qadf_native PRIVATE BENCHMARK_AVAILABLE)
    endif()
endif()
