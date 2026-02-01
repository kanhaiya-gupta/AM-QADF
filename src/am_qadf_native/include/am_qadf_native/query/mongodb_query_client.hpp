#ifndef AM_QADF_NATIVE_QUERY_MONGODB_QUERY_CLIENT_HPP
#define AM_QADF_NATIVE_QUERY_MONGODB_QUERY_CLIENT_HPP

#include "query_result.hpp"
#include <mongocxx/client.hpp>
#include <mongocxx/database.hpp>
#include <mongocxx/collection.hpp>
#include <bson/bson.h>
#include <string>
#include <vector>
#include <memory>
#include <array>
#include <limits>

namespace am_qadf_native {
namespace query {

// MongoDB query client
class MongoDBQueryClient {
private:
    mongocxx::client client_;
    mongocxx::database db_;
    
public:
    MongoDBQueryClient(const std::string& uri, const std::string& db_name);
    
    // Query laser monitoring data (LBD - Laser Beam Diagnostics)
    QueryResult queryLaserMonitoringData(
        const std::string& model_id,
        int layer_start = -1,
        int layer_end = -1,
        const std::array<float, 3>& bbox_min = {std::numeric_limits<float>::lowest(), 
                                                  std::numeric_limits<float>::lowest(), 
                                                  std::numeric_limits<float>::lowest()},
        const std::array<float, 3>& bbox_max = {std::numeric_limits<float>::max(), 
                                                  std::numeric_limits<float>::max(), 
                                                  std::numeric_limits<float>::max()}
    );
    
    // Query ISPM_Thermal (In-Situ Process Monitoring - Thermal) data
    QueryResult queryISPMThermal(
        const std::string& model_id,
        float time_start = 0.0f,
        float time_end = 0.0f,
        int layer_start = -1,
        int layer_end = -1,
        const std::array<float, 3>& bbox_min = {std::numeric_limits<float>::lowest(), 
                                                  std::numeric_limits<float>::lowest(), 
                                                  std::numeric_limits<float>::lowest()},
        const std::array<float, 3>& bbox_max = {std::numeric_limits<float>::max(), 
                                                  std::numeric_limits<float>::max(), 
                                                  std::numeric_limits<float>::max()}
    );
    
    // Query ISPM_Optical (In-Situ Process Monitoring - Optical) data
    QueryResult queryISPMOptical(
        const std::string& model_id,
        float time_start = 0.0f,
        float time_end = 0.0f,
        int layer_start = -1,
        int layer_end = -1,
        const std::array<float, 3>& bbox_min = {std::numeric_limits<float>::lowest(), 
                                                  std::numeric_limits<float>::lowest(), 
                                                  std::numeric_limits<float>::lowest()},
        const std::array<float, 3>& bbox_max = {std::numeric_limits<float>::max(), 
                                                  std::numeric_limits<float>::max(), 
                                                  std::numeric_limits<float>::max()}
    );
    
    // Query ISPM_Acoustic (In-Situ Process Monitoring - Acoustic) data
    QueryResult queryISPMAcoustic(
        const std::string& model_id,
        float time_start = 0.0f,
        float time_end = 0.0f,
        int layer_start = -1,
        int layer_end = -1,
        const std::array<float, 3>& bbox_min = {std::numeric_limits<float>::lowest(), 
                                                  std::numeric_limits<float>::lowest(), 
                                                  std::numeric_limits<float>::lowest()},
        const std::array<float, 3>& bbox_max = {std::numeric_limits<float>::max(), 
                                                  std::numeric_limits<float>::max(), 
                                                  std::numeric_limits<float>::max()}
    );
    
    // Query ISPM_Strain (In-Situ Process Monitoring - Strain) data
    QueryResult queryISPMStrain(
        const std::string& model_id,
        float time_start = 0.0f,
        float time_end = 0.0f,
        int layer_start = -1,
        int layer_end = -1,
        const std::array<float, 3>& bbox_min = {std::numeric_limits<float>::lowest(), 
                                                  std::numeric_limits<float>::lowest(), 
                                                  std::numeric_limits<float>::lowest()},
        const std::array<float, 3>& bbox_max = {std::numeric_limits<float>::max(), 
                                                  std::numeric_limits<float>::max(), 
                                                  std::numeric_limits<float>::max()}
    );
    
    // Query ISPM_Plume (In-Situ Process Monitoring - Plume) data
    QueryResult queryISPMPlume(
        const std::string& model_id,
        float time_start = 0.0f,
        float time_end = 0.0f,
        int layer_start = -1,
        int layer_end = -1,
        const std::array<float, 3>& bbox_min = {std::numeric_limits<float>::lowest(), 
                                                  std::numeric_limits<float>::lowest(), 
                                                  std::numeric_limits<float>::lowest()},
        const std::array<float, 3>& bbox_max = {std::numeric_limits<float>::max(), 
                                                  std::numeric_limits<float>::max(), 
                                                  std::numeric_limits<float>::max()}
    );
    
    // Query CT scan metadata
    QueryResult queryCTScan(const std::string& model_id);
    
    // Query hatching data (fast C++ implementation)
    QueryResult queryHatchingData(
        const std::string& model_id,
        int layer_start = -1,
        int layer_end = -1,
        const std::array<float, 3>& bbox_min = {std::numeric_limits<float>::lowest(), 
                                                  std::numeric_limits<float>::lowest(), 
                                                  std::numeric_limits<float>::lowest()},
        const std::array<float, 3>& bbox_max = {std::numeric_limits<float>::max(), 
                                                  std::numeric_limits<float>::max(), 
                                                  std::numeric_limits<float>::max()}
    );
    
    // Get CT scan file path from metadata
    std::string getCTScanFilePath(const std::string& model_id, const std::string& format = "dicom");
    
private:
    // Helper: Parse MongoDB document to QueryResult
    void parseDocument(const bsoncxx::document::view& doc, QueryResult& result);
    
    // Helper: Parse ISPM_Thermal-specific MongoDB document (extracts all ISPM_Thermal fields in C++)
    void parseISPMThermalDocument(const bsoncxx::document::view& doc, QueryResult& result);
    
    // Helper: Parse ISPM_Optical-specific MongoDB document (extracts all ISPM_Optical fields in C++)
    void parseISPMOpticalDocument(const bsoncxx::document::view& doc, QueryResult& result);
    
    // Helper: Parse ISPM_Acoustic-specific MongoDB document (extracts all ISPM_Acoustic fields in C++)
    void parseISPMAcousticDocument(const bsoncxx::document::view& doc, QueryResult& result);
    
    // Helper: Parse ISPM_Strain-specific MongoDB document (extracts all ISPM_Strain fields in C++)
    void parseISPMStrainDocument(const bsoncxx::document::view& doc, QueryResult& result);
    
    // Helper: Parse ISPM_Plume-specific MongoDB document (extracts all ISPM_Plume fields in C++)
    void parseISPMPlumeDocument(const bsoncxx::document::view& doc, QueryResult& result);
    
    // Helper: Extract coordinates from document (handles multiple field name variations)
    std::array<float, 3> extractCoordinates(const bsoncxx::document::view& doc);
    
    // Helper: Extract layer index from document (handles multiple field name variations)
    int extractLayerIndex(const bsoncxx::document::view& doc);
    
    // Helper: Extract float value from document (handles multiple field name variations)
    float extractFloatValue(const bsoncxx::document::view& doc, 
                           const std::vector<std::string>& field_names,
                           float default_value = 0.0f);
    
    // Helper: Extract int value from document (handles multiple field name variations)
    int extractIntValue(const bsoncxx::document::view& doc, 
                       const std::vector<std::string>& field_names,
                       int default_value = 0);
    
    // Helper: Extract string value from document (handles multiple field name variations)
    std::string extractStringValue(const bsoncxx::document::view& doc,
                                  const std::vector<std::string>& field_names,
                                  const std::string& default_value = "");
};

} // namespace query
} // namespace am_qadf_native

#endif // AM_QADF_NATIVE_QUERY_MONGODB_QUERY_CLIENT_HPP
