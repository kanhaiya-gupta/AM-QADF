#include "am_qadf_native/query/ct_scan_query.hpp"
#include <openvdb/openvdb.h>
#include <string>
#include <memory>

namespace am_qadf_native {
namespace query {

QueryResult CTScanQuery::queryMetadata(const std::string& model_id) {
    return client_->queryCTScan(model_id);
}

FloatGridPtr CTScanQuery::readVolume(
    const std::string& model_id,
    const std::string& format
) {
    // Get file path from metadata
    std::string file_path = client_->getCTScanFilePath(model_id, format);
    
    if (file_path.empty()) {
        return FloatGrid::create();  // No file path found
    }
    
    // Read volume based on format
    if (format == "dicom" || format == "dcm") {
        return reader_->readDICOMSeries(file_path);
    } else if (format == "nifti" || format == "nii") {
        return reader_->readNIfTI(file_path);
    } else if (format == "tiff" || format == "tif") {
        return reader_->readTIFFStack(file_path);
    } else {
        // Unknown format - return empty grid
        return FloatGrid::create();
    }
}

FloatGridPtr CTScanQuery::queryAndRead(const std::string& model_id) {
    // Query metadata and read volume in one operation
    return readVolume(model_id, "dicom");  // Default to DICOM
}

} // namespace query
} // namespace am_qadf_native
