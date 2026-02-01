#ifndef AM_QADF_NATIVE_QUERY_CT_SCAN_QUERY_HPP
#define AM_QADF_NATIVE_QUERY_CT_SCAN_QUERY_HPP

#include "mongodb_query_client.hpp"
#include "ct_image_reader.hpp"
#include "query_result.hpp"
#include <openvdb/Grid.h>
#include <memory>

namespace am_qadf_native {
namespace query {

using FloatGrid = openvdb::FloatGrid;
using FloatGridPtr = FloatGrid::Ptr;

// CT scan query utilities
class CTScanQuery {
private:
    MongoDBQueryClient* client_;
    CTImageReader* reader_;
    
public:
    CTScanQuery(MongoDBQueryClient* client, CTImageReader* reader) 
        : client_(client), reader_(reader) {}
    
    // Query CT scan metadata
    QueryResult queryMetadata(const std::string& model_id);
    
    // Read CT scan volume (DICOM, NIfTI, etc.)
    FloatGridPtr readVolume(const std::string& model_id, const std::string& format = "dicom");
    
    // Query and read in one operation
    FloatGridPtr queryAndRead(const std::string& model_id);
};

} // namespace query
} // namespace am_qadf_native

#endif // AM_QADF_NATIVE_QUERY_CT_SCAN_QUERY_HPP
