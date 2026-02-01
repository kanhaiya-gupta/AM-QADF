#include "am_qadf_native/query/laser_monitoring_query.hpp"
#include <mongocxx/collection.hpp>
#include <bsoncxx/builder/stream/document.hpp>
#include <string>
#include <array>
#include <limits>

namespace am_qadf_native {
namespace query {

QueryResult LaserMonitoringQuery::queryByLayer(
    const std::string& model_id,
    int layer_start,
    int layer_end
) {
    return client_->queryLaserMonitoringData(model_id, layer_start, layer_end);
}

QueryResult LaserMonitoringQuery::queryByTime(
    const std::string& model_id,
    float time_start,
    float time_end
) {
    QueryResult result;
    result.model_id = model_id;
    result.signal_type = "laser_monitoring_data";
    
    // Build query with time range
    bsoncxx::builder::stream::document filter_builder;
    filter_builder << "model_id" << model_id;
    
    if (time_start >= 0.0f && time_end >= 0.0f && time_end >= time_start) {
        filter_builder << "timestamp" << bsoncxx::builder::stream::open_document
                      << "$gte" << time_start
                      << "$lte" << time_end
                      << bsoncxx::builder::stream::close_document;
    }
    
    auto filter = filter_builder.extract();
    
    // Execute query - need access to client's database
    // Since we only have MongoDBQueryClient pointer, we'll need to add a method
    // or use the existing queryLaserMonitoringData and filter by time in post-processing
    // For now, use a workaround: query all layers and filter by time
    
    // Get all layers (wide range) and filter by time
    // This is not ideal but works with current API
    result = client_->queryLaserMonitoringData(model_id, 0, 10000);  // Large range
    
    // Filter results by time range
    if (time_start >= 0.0f && time_end >= 0.0f) {
        QueryResult filtered_result;
        filtered_result.model_id = result.model_id;
        filtered_result.signal_type = result.signal_type;
        
        for (size_t i = 0; i < result.points.size(); ++i) {
            if (i < result.timestamps.size()) {
                float ts = result.timestamps[i];
                if (ts >= time_start && ts <= time_end) {
                    filtered_result.points.push_back(result.points[i]);
                    if (i < result.values.size()) {
                        filtered_result.values.push_back(result.values[i]);
                    }
                    filtered_result.timestamps.push_back(ts);
                    if (i < result.layers.size()) {
                        filtered_result.layers.push_back(result.layers[i]);
                    }
                    // Signals come from laser_temporal_data (structured data)
                    if (i < result.laser_temporal_data.size()) {
                        filtered_result.laser_temporal_data.push_back(result.laser_temporal_data[i]);
                    }
                }
            }
        }
        
        return filtered_result;
    }
    
    return result;
}

QueryResult LaserMonitoringQuery::queryBySpatialRegion(
    const std::string& model_id,
    float bbox_min[3],
    float bbox_max[3]
) {
    QueryResult result;
    result.model_id = model_id;
    result.signal_type = "laser_monitoring_data";
    
    // Build query with spatial bounding box
    bsoncxx::builder::stream::document filter_builder;
    filter_builder << "model_id" << model_id;
    
    // Add spatial filter using MongoDB's $geoWithin or coordinate ranges
    // For coordinate-based queries, we can filter by x, y, z ranges
    filter_builder << "x" << bsoncxx::builder::stream::open_document
                  << "$gte" << bbox_min[0]
                  << "$lte" << bbox_max[0]
                  << bsoncxx::builder::stream::close_document;
    filter_builder << "y" << bsoncxx::builder::stream::open_document
                  << "$gte" << bbox_min[1]
                  << "$lte" << bbox_max[1]
                  << bsoncxx::builder::stream::close_document;
    filter_builder << "z" << bsoncxx::builder::stream::open_document
                  << "$gte" << bbox_min[2]
                  << "$lte" << bbox_max[2]
                  << bsoncxx::builder::stream::close_document;
    
    auto filter = filter_builder.extract();
    
    // Execute query - similar workaround as time query
    // Query all layers and filter spatially
    result = client_->queryLaserMonitoringData(model_id, 0, 10000);
    
    // Filter results by spatial bounding box
    QueryResult filtered_result;
    filtered_result.model_id = result.model_id;
    filtered_result.signal_type = result.signal_type;
    
    for (size_t i = 0; i < result.points.size(); ++i) {
        const auto& pt = result.points[i];
        if (pt[0] >= bbox_min[0] && pt[0] <= bbox_max[0] &&
            pt[1] >= bbox_min[1] && pt[1] <= bbox_max[1] &&
            pt[2] >= bbox_min[2] && pt[2] <= bbox_max[2]) {
            filtered_result.points.push_back(pt);
            if (i < result.values.size()) {
                filtered_result.values.push_back(result.values[i]);
            }
            if (i < result.timestamps.size()) {
                filtered_result.timestamps.push_back(result.timestamps[i]);
            }
            if (i < result.layers.size()) {
                filtered_result.layers.push_back(result.layers[i]);
            }
            // Signals come from laser_temporal_data (structured data)
            if (i < result.laser_temporal_data.size()) {
                filtered_result.laser_temporal_data.push_back(result.laser_temporal_data[i]);
            }
        }
    }
    
    return filtered_result;
}

} // namespace query
} // namespace am_qadf_native
