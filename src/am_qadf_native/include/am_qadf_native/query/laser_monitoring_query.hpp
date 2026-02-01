#ifndef AM_QADF_NATIVE_QUERY_LASER_MONITORING_QUERY_HPP
#define AM_QADF_NATIVE_QUERY_LASER_MONITORING_QUERY_HPP

#include "mongodb_query_client.hpp"
#include "query_result.hpp"

namespace am_qadf_native {
namespace query {

// Laser monitoring data query utilities (LBD - Laser Beam Diagnostics)
class LaserMonitoringQuery {
private:
    MongoDBQueryClient* client_;
    
public:
    LaserMonitoringQuery(MongoDBQueryClient* client) : client_(client) {}
    
    QueryResult queryByLayer(const std::string& model_id, int layer_start, int layer_end);
    QueryResult queryByTime(const std::string& model_id, float time_start, float time_end);
    QueryResult queryBySpatialRegion(const std::string& model_id, 
                                     float bbox_min[3], float bbox_max[3]);
};

} // namespace query
} // namespace am_qadf_native

#endif // AM_QADF_NATIVE_QUERY_LASER_MONITORING_QUERY_HPP
