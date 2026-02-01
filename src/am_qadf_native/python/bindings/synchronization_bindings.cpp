#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include "am_qadf_native/synchronization/grid_spatial_alignment.hpp"
#include "am_qadf_native/synchronization/grid_temporal_alignment.hpp"
#include "am_qadf_native/synchronization/grid_synchronizer.hpp"
#include "am_qadf_native/synchronization/point_transformation_estimate.hpp"
#include "am_qadf_native/synchronization/point_transformation_validate.hpp"
#include "am_qadf_native/synchronization/point_transform.hpp"
#include "am_qadf_native/synchronization/point_bounds.hpp"
#include "am_qadf_native/synchronization/point_coordinate_transform.hpp"
#include "am_qadf_native/synchronization/point_transformation_sampling.hpp"
#include "am_qadf_native/synchronization/point_temporal_alignment.hpp"
#include <Eigen/Dense>

namespace py = pybind11;
using namespace am_qadf_native::synchronization;

void bind_synchronization(py::module& m) {
    // SpatialAlignment
    py::class_<SpatialAlignment>(m, "SpatialAlignment")
        .def(py::init<>())
        .def("align", &SpatialAlignment::align, 
             py::arg("source_grid"), py::arg("target_grid"), py::arg("method") = "trilinear")
        .def("align_with_transform", &SpatialAlignment::alignWithTransform)
        .def("transforms_match", &SpatialAlignment::transformsMatch,
             py::arg("grid1"), py::arg("grid2"), py::arg("tolerance") = 1e-6,
             "Check if two grids have the same transform")
        .def("get_transform_matrix", &SpatialAlignment::getTransformMatrix,
             py::arg("grid"),
             "Get the transform matrix of a grid as a 4x4 array")
        .def("get_world_bounding_box", &SpatialAlignment::getWorldBoundingBox,
             py::arg("grid"),
             "Get world-space bounding box as [min_x, min_y, min_z, max_x, max_y, max_z]")
        .def("align_to_bounding_box", &SpatialAlignment::alignToBoundingBox,
             py::arg("source_grid"), py::arg("target_grid"),
             py::arg("unified_min_x"), py::arg("unified_min_y"), py::arg("unified_min_z"),
             py::arg("unified_max_x"), py::arg("unified_max_y"), py::arg("unified_max_z"),
             py::arg("method") = "trilinear",
             "Resample source grid into unified bounding box with target transform (moves data spatially)");
    
    // TemporalAlignment
    py::class_<TemporalAlignment>(m, "TemporalAlignment")
        .def(py::init<>())
        .def("synchronize_temporal", &TemporalAlignment::synchronizeTemporal,
             py::arg("grids"), py::arg("temporal_window") = 0.1f, py::arg("layer_tolerance") = 1);
    
    // GridSynchronizer
    py::class_<GridSynchronizer>(m, "GridSynchronizer")
        .def(py::init<>())
        .def("synchronize", &GridSynchronizer::synchronize,
             py::arg("source_grids"), py::arg("reference_grid"),
             py::arg("timestamps"), py::arg("layer_indices"),
             py::arg("temporal_window") = 0.1f, py::arg("layer_tolerance") = 1)
        .def("synchronize_spatial", &GridSynchronizer::synchronizeSpatial)
        .def("synchronize_temporal", &GridSynchronizer::synchronizeTemporal);
    
#ifdef EIGEN_AVAILABLE
    // Transformation sampling: bbox corners → 56 triplets (fit on 3, validate on 8)
    m.def("enumerate_triplet_samples_from_8", &TransformationSampling::enumerateTripletSamplesFrom8,
          py::arg("source_8"), py::arg("target_8"),
          "Enumerate all C(8,3)=56 triplets from 8 bbox corners; returns list of (source_3x3, target_3x3) pairs.");

    // Point temporal alignment: align point sets by layer for per-layer fusion
    py::class_<LayerAlignmentResult>(m, "LayerAlignmentResult")
        .def_readwrite("unique_layers", &LayerAlignmentResult::unique_layers,
                       "Sorted list of layer indices that appear in any source")
        .def_readwrite("indices_per_layer_per_source", &LayerAlignmentResult::indices_per_layer_per_source,
                       "[layer_idx][source_idx] = list of row indices into that source's points for that layer");

    py::class_<PointTemporalAlignment>(m, "PointTemporalAlignment")
        .def(py::init<>())
        .def("align_sources_by_layer", &PointTemporalAlignment::alignSourcesByLayer,
             py::arg("points_per_source"), py::arg("layer_indices_per_source"),
             "Align multiple sources by layer; returns unique_layers and indices_per_layer_per_source for slicing.")
        .def("align_points_by_layer", &am_qadf_native::synchronization::align_points_by_layer_from_dicts,
             py::arg("transformed_points"), py::arg("layer_indices_per_source"),
             py::arg("source_order") = py::none(),
             "Align by layer from dicts; order/validation/conversion in C++. Returns (LayerAlignmentResult, source_order_used).")
        .def("align_sources_by_layer_from_arrays", [](PointTemporalAlignment& self,
             const py::list& layer_arrays) {
            const size_t n = static_cast<size_t>(py::len(layer_arrays));
            std::vector<const int*> ptrs(n);
            std::vector<size_t> sizes(n);
            for (size_t s = 0; s < n; ++s) {
                py::array_t<int32_t> arr = py::cast<py::array_t<int32_t>>(layer_arrays[s]);
                arr = py::array_t<int32_t>::ensure(arr);
                ptrs[s] = arr.data();
                sizes[s] = static_cast<size_t>(arr.size());
            }
            return self.alignSourcesByLayerFromLayerArraysOnly(ptrs, sizes);
        }, py::arg("layer_arrays"),
           "Zero-copy: align by layer using contiguous layer index arrays (numpy int32). No points passed; use for large data.")
        .def("group_indices_by_layer", [](const PointTemporalAlignment& self, const std::vector<int>& layer_indices) {
            std::vector<int> unique_layers;
            std::vector<std::vector<int>> indices_per_layer;
            self.groupIndicesByLayer(layer_indices, unique_layers, indices_per_layer);
            return py::make_tuple(unique_layers, indices_per_layer);
        }, py::arg("layer_indices"),
           "Group a single source's points by layer; returns (unique_layers, indices_per_layer).");
#endif

    // ============================================
    // TransformationComputer
    // ============================================
    
    // RANSACResult struct
    py::class_<RANSACResult>(m, "RANSACResult")
        .def_readwrite("transformation", &RANSACResult::transformation)
        .def_readwrite("inlier_indices", &RANSACResult::inlier_indices)
        .def_readwrite("inlier_source_points", &RANSACResult::inlier_source_points)
        .def_readwrite("inlier_target_points", &RANSACResult::inlier_target_points)
        .def_readwrite("num_inliers", &RANSACResult::num_inliers)
        .def_readwrite("confidence", &RANSACResult::confidence);
    
    // TransformationQuality struct
    py::class_<TransformationQuality>(m, "TransformationQuality")
        .def_readwrite("rms_error", &TransformationQuality::rms_error)
        .def_readwrite("alignment_quality", &TransformationQuality::alignment_quality)
        .def_readwrite("confidence", &TransformationQuality::confidence)
        .def_readwrite("max_error", &TransformationQuality::max_error)
        .def_readwrite("mean_error", &TransformationQuality::mean_error);

    // BboxFitCandidate: per-fit error for one (permutation_index, triplet_index) in bbox corner fitting
    py::class_<BboxFitCandidate>(m, "BboxFitCandidate")
        .def_readwrite("permutation_index", &BboxFitCandidate::permutation_index)
        .def_readwrite("triplet_index", &BboxFitCandidate::triplet_index)
        .def_readwrite("max_error", &BboxFitCandidate::max_error)
        .def_readwrite("mean_error", &BboxFitCandidate::mean_error)
        .def_readwrite("rms_error", &BboxFitCandidate::rms_error);

    // ScaleTranslationRotation: decomposition of 4x4 similarity matrix
    py::class_<ScaleTranslationRotation>(m, "ScaleTranslationRotation")
        .def_readwrite("scale", &ScaleTranslationRotation::scale)
        .def_readwrite("tx", &ScaleTranslationRotation::tx)
        .def_readwrite("ty", &ScaleTranslationRotation::ty)
        .def_readwrite("tz", &ScaleTranslationRotation::tz)
        .def_readwrite("rot_x_deg", &ScaleTranslationRotation::rot_x_deg)
        .def_readwrite("rot_y_deg", &ScaleTranslationRotation::rot_y_deg)
        .def_readwrite("rot_z_deg", &ScaleTranslationRotation::rot_z_deg);

    m.def("decompose_similarity_transform", &decomposeSimilarityTransform,
          py::arg("matrix_4x4"),
          "Decompose 4x4 similarity matrix into scale, translation (tx,ty,tz), rotation (Euler ZYX degrees).");
    
    // TransformationComputer class
    py::class_<TransformationComputer>(m, "TransformationComputer")
        .def(py::init<>())
        .def("compute_with_ransac", &TransformationComputer::computeWithRANSAC,
             py::arg("source_points"), py::arg("target_points"),
             py::arg("threshold") = 0.005,
             py::arg("min_sets") = 10,
             py::arg("consensus_threshold") = 7,
             py::arg("consensus_tolerance") = 1e-6,
             py::arg("max_iterations") = 1000,
             py::arg("confidence") = 0.99,
             py::arg("min_inliers") = 3,
             "Compute transformation using RANSAC (multi-set approach)")
        .def("compute_optimal_transformation", &TransformationComputer::computeOptimalTransformation,
             py::arg("source_points"), py::arg("target_points"),
             py::arg("method") = "kabsch_umeyama",
             "Compute optimal transformation using Kabsch + Umeyama")
        .def("compute_robust_transformation", [](TransformationComputer& self,
             const Eigen::MatrixXd& source_points,
             const Eigen::MatrixXd& target_points) {
            TransformationQuality quality;
            Eigen::Matrix4d transform = self.computeRobustTransformation(
                source_points, target_points, quality
            );
            return std::make_tuple(transform, quality);
        }, py::arg("source_points"), py::arg("target_points"),
           "Compute robust transformation with quality metrics")
        .def("compute_kabsch_rotation", &TransformationComputer::computeKabschRotation,
             py::arg("source_points"), py::arg("target_points"),
             py::arg("source_centroid"), py::arg("target_centroid"),
             "Compute optimal rotation using Kabsch algorithm")
        .def("compute_umeyama_scaling", &TransformationComputer::computeUmeyamaScaling,
             py::arg("source_points"), py::arg("target_points"),
             py::arg("rotation"), py::arg("source_centroid"), py::arg("target_centroid"),
             "Compute uniform scaling using Umeyama's method")
        .def("refine_transformation", &TransformationComputer::refineTransformation,
             py::arg("initial_transform"), py::arg("source_points"), py::arg("target_points"),
             "Refine existing transformation using data")
        .def("compute_quality_metrics", &TransformationComputer::computeQualityMetrics,
             py::arg("source_points"), py::arg("target_points"), py::arg("transformation"),
             "Compute quality metrics for transformation")
        .def("compute_transformation_from_bbox_corners", [](TransformationComputer& self,
             const Eigen::MatrixXd& source_corners,
             const Eigen::MatrixXd& reference_corners) {
            TransformationQuality quality;
            Eigen::Matrix4d transform = self.computeTransformationFromBboxCorners(
                source_corners, reference_corners, &quality, nullptr, nullptr
            );
            return std::make_tuple(transform, quality);
        }, py::arg("source_corners"), py::arg("reference_corners"),
           "Bbox corners (8x3 each): 24 permutations × 56 triplets — fit on 3, validate on 8; return (transform, quality) with least error.")
        .def("compute_transformation_from_bbox_corners_with_fit_errors", [](TransformationComputer& self,
             const Eigen::MatrixXd& source_corners,
             const Eigen::MatrixXd& reference_corners) {
            TransformationQuality quality;
            std::vector<BboxFitCandidate> all_fits;
            Eigen::MatrixXd best_ref_corners(8, 3);
            Eigen::Matrix4d transform = self.computeTransformationFromBboxCorners(
                source_corners, reference_corners, &quality, &all_fits, &best_ref_corners
            );
            return std::make_tuple(transform, quality, all_fits, best_ref_corners);
        }, py::arg("source_corners"), py::arg("reference_corners"),
           "Same as compute_transformation_from_bbox_corners but also returns (fit_errors, best_ref_corners). Validate using best_ref_corners (reordered by best permutation).");
    
    // ============================================
    // TransformationValidator
    // ============================================
    
    // BoundingBox struct
    py::class_<BoundingBox>(m, "BoundingBox")
        .def(py::init<>())
        .def(py::init<double, double, double, double, double, double>(),
             py::arg("min_x"), py::arg("min_y"), py::arg("min_z"),
             py::arg("max_x"), py::arg("max_y"), py::arg("max_z"))
        .def_readwrite("min_x", &BoundingBox::min_x)
        .def_readwrite("min_y", &BoundingBox::min_y)
        .def_readwrite("min_z", &BoundingBox::min_z)
        .def_readwrite("max_x", &BoundingBox::max_x)
        .def_readwrite("max_y", &BoundingBox::max_y)
        .def_readwrite("max_z", &BoundingBox::max_z)
        .def("expand", py::overload_cast<const Eigen::Vector3d&>(&BoundingBox::expand))
        .def("expand", py::overload_cast<const BoundingBox&>(&BoundingBox::expand))
        .def("contains", &BoundingBox::contains)
        .def("width", &BoundingBox::width)
        .def("height", &BoundingBox::height)
        .def("depth", &BoundingBox::depth)
        .def("center", &BoundingBox::center)
        .def("size", &BoundingBox::size)
        .def("is_valid", &BoundingBox::isValid)
        .def("corners", &BoundingBox::corners,
             "Return 8 corner points in fixed order for bbox correspondence (8x3 matrix).");
    
    // ValidationResult struct
    py::class_<ValidationResult>(m, "ValidationResult")
        .def(py::init<>())
        .def_readwrite("is_valid", &ValidationResult::isValid)
        .def_readwrite("max_error", &ValidationResult::max_error)
        .def_readwrite("mean_error", &ValidationResult::mean_error)
        .def_readwrite("rms_error", &ValidationResult::rms_error)
        .def_readwrite("errors", &ValidationResult::errors)
        .def_readwrite("warnings", &ValidationResult::warnings);

    // BboxCorrespondenceValidation struct (corners + centre, 9 pairs)
    py::class_<BboxCorrespondenceValidation>(m, "BboxCorrespondenceValidation")
        .def(py::init<>())
        .def_readwrite("mean_distance", &BboxCorrespondenceValidation::mean_distance)
        .def_readwrite("max_distance", &BboxCorrespondenceValidation::max_distance)
        .def_readwrite("num_pairs", &BboxCorrespondenceValidation::num_pairs);
    
    // TransformationValidator class
    py::class_<TransformationValidator>(m, "TransformationValidator")
        .def(py::init<>())
        .def("validate", [](TransformationValidator& self,
             const CoordinateSystem& source_system,
             const CoordinateSystem& target_system,
             const std::vector<Eigen::Vector3d>& sample_points,
             double tolerance = 1e-9) {
            return self.validate(source_system, target_system, sample_points, tolerance);
        }, py::arg("source_system"), py::arg("target_system"), py::arg("sample_points"),
           py::arg("tolerance") = 1e-9,
           "Validate transformation with sample points")
        .def("validate_with_matrix", &TransformationValidator::validateWithMatrix,
             py::arg("source_points"), py::arg("transformation"), py::arg("target_points"),
             py::arg("tolerance") = 1e-9,
             "Validate transformation with pre-computed matrix")
        .def("validate_with_quality_metrics", &TransformationValidator::validateWithQualityMetrics,
             py::arg("source_points"), py::arg("transformation"), py::arg("target_points"),
             py::arg("tolerance") = 1e-9,
             "Validate with quality metrics")
        .def("validate_round_trip", &TransformationValidator::validateRoundTrip,
             py::arg("points"), py::arg("forward_transform"), py::arg("inverse_transform"),
             py::arg("tolerance") = 1e-9,
             "Round-trip validation")
        .def("validate_geometric_consistency", &TransformationValidator::validateGeometricConsistency,
             py::arg("points"), py::arg("transform"),
             "Geometric consistency validation")
        .def("validate_bounds", &TransformationValidator::validateBounds,
             py::arg("transformed_points"), py::arg("expected_bounds"),
             py::arg("max_out_of_bounds_ratio") = 0.01,
             "Statistical bounds validation")
        .def("validate_bbox_corners_and_centre", &TransformationValidator::validateBboxCornersAndCentre,
             py::arg("source_corners"), py::arg("reference_corners"), py::arg("transformation"),
             "Point correspondence check on 8 bbox corners + 1 centre (9 known pairs); returns mean/max distance.");
    
    // ============================================
    // PointTransformer
    // ============================================
    
    py::class_<PointTransformer>(m, "PointTransformer")
        .def(py::init<>())
        .def("transform", [](PointTransformer& self,
             const Eigen::MatrixXd& points,
             const CoordinateSystem& source_system,
             const CoordinateSystem& target_system) {
            return self.transform(points, source_system, target_system);
        }, py::arg("points"), py::arg("source_system"), py::arg("target_system"),
           "Transform points from source to target coordinate system")
        .def("transform_with_matrix", &PointTransformer::transformWithMatrix,
             py::arg("points"), py::arg("transform_matrix"),
             "Transform with pre-validated transformation matrix")
        .def("get_transform_matrix", &PointTransformer::getTransformMatrix,
             py::arg("source_system"), py::arg("target_system"),
             "Get transformation matrix for reuse");
    
    // ============================================
    // UnifiedBoundsComputer
    // ============================================
    
    py::class_<UnifiedBoundsComputer>(m, "UnifiedBoundsComputer")
        .def(py::init<>())
        .def("compute_bounds_from_points", &UnifiedBoundsComputer::computeBoundsFromPoints,
             py::arg("points"),
             "Compute bounding box from point set (rows=points, cols=x,y,z).")
        .def("compute_union_bounds", &UnifiedBoundsComputer::computeUnionBounds,
             py::arg("transformed_point_sets"),
             "Compute union bounds from transformed points")
        .def("compute_incremental", [](UnifiedBoundsComputer& self,
             const std::vector<Eigen::MatrixXd>& point_sets,
             const std::vector<CoordinateSystem>& source_systems,
             const CoordinateSystem& target_system) {
            return self.computeIncremental(point_sets, source_systems, target_system);
        }, py::arg("point_sets"), py::arg("source_systems"), py::arg("target_system"),
           "Incremental bounds expansion")
        .def("add_padding", &UnifiedBoundsComputer::addPadding,
             py::arg("bounds"), py::arg("padding"),
             "Add padding to bounds")
        .def("add_percentage_padding", &UnifiedBoundsComputer::addPercentagePadding,
             py::arg("bounds"), py::arg("padding_percent"),
             "Add percentage-based padding");
    
    // ============================================
    // CoordinateSystem and CoordinateTransformer
    // ============================================
    
    // CoordinateSystem
    py::class_<CoordinateSystem>(m, "CoordinateSystem")
        .def(py::init<>())
        .def_readwrite("origin", &CoordinateSystem::origin)
        .def_readwrite("rotation_euler", &CoordinateSystem::rotation_euler)
        .def_readwrite("scale", &CoordinateSystem::scale)
        .def_readwrite("rotation_axis", &CoordinateSystem::rotation_axis)
        .def_readwrite("rotation_angle", &CoordinateSystem::rotation_angle)
        .def_readwrite("use_axis_angle", &CoordinateSystem::use_axis_angle);
    
    // CoordinateTransformer
    py::class_<CoordinateTransformer>(m, "CoordinateTransformer")
        .def(py::init<>())
        .def("transform_point", &CoordinateTransformer::transformPoint,
             py::arg("point"), py::arg("from_system"), py::arg("to_system"),
             "Transform single point between coordinate systems")
        .def("transform_points", &CoordinateTransformer::transformPoints,
             py::arg("points"), py::arg("from_system"), py::arg("to_system"),
             "Transform batch of points (vectorized)")
        .def("build_transform_matrix", &CoordinateTransformer::buildTransformMatrix,
             py::arg("system"),
             "Build 4x4 homogeneous transformation matrix from coordinate system")
        .def("build_inverse_transform_matrix", &CoordinateTransformer::buildInverseTransformMatrix,
             py::arg("system"),
             "Build inverse transformation matrix")
        .def("rotation_matrix_from_euler", &CoordinateTransformer::rotationMatrixFromEuler,
             py::arg("rx"), py::arg("ry"), py::arg("rz"),
             "Compute 3x3 rotation matrix from Euler angles (ZYX order)")
        .def("rotation_matrix_from_axis_angle", &CoordinateTransformer::rotationMatrixFromAxisAngle,
             py::arg("axis"), py::arg("angle"),
             "Compute 3x3 rotation matrix from axis-angle representation")
        .def("validate_coordinate_system", &CoordinateTransformer::validateCoordinateSystem,
             py::arg("system"),
             "Validate coordinate system");
}
