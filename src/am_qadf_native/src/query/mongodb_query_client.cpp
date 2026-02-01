#include "am_qadf_native/query/mongodb_query_client.hpp"
#include "am_qadf_native/io/mongocxx_instance.hpp"
#include <mongocxx/uri.hpp>
#include <bson/bson.h>
#include <bsoncxx/builder/stream/document.hpp>
#include <bsoncxx/json.hpp>
#include <bsoncxx/types.hpp>
#include <algorithm>
#include <array>
#include <string>
#include <cmath>  // For std::sqrt in ISPM calculations
#include <limits>  // For std::numeric_limits
#include <vector>
#include <ctime>
#include <cstdlib>

namespace am_qadf_native {
namespace query {

namespace {
// Trim leading/trailing whitespace in place; returns substring length to use.
size_t trimTimestampString(const std::string& s, size_t* start_out) {
    size_t start = 0;
    while (start < s.size() && (s[start] == ' ' || s[start] == '\t')) ++start;
    size_t end = s.size();
    while (end > start && (s[end - 1] == ' ' || s[end - 1] == '\t' || s[end - 1] == 'Z' || s[end - 1] == 'z')) --end;
    *start_out = start;
    return end > start ? end - start : 0;
}

// Parse various timestamp string formats to Unix seconds (float). Supports:
// - Unix numeric: "1736932200", "1736932200.123"
// - ISO 8601 with T: "2025-01-15T10:30:00", "2025-01-15T10:30:00.123456", "2025-01-15T10:30:00Z"
// - ISO 8601 with space: "2025-01-15 10:30:00", "2025-01-15 10:30:00.123"
// - ISO with slash: "2025/01/15 10:30:00", "2025/01/15T10:30:00"
float parseTimestampStringToUnix(const std::string& s) {
    if (s.empty()) return 0.0f;
    size_t start = 0;
    size_t len = trimTimestampString(s, &start);
    if (len == 0) return 0.0f;
    const char* p = s.c_str() + start;

    // 1) Pure numeric (Unix seconds or float)
    char* end = nullptr;
    float num = std::strtof(p, &end);
    if (end && end > p) {
        size_t consumed = static_cast<size_t>(end - p);
        bool all_numeric = true;
        for (size_t i = 0; i < consumed; ++i) {
            char c = p[i];
            if (c != '.' && (c < '0' || c > '9') && c != '+' && c != '-') { all_numeric = false; break; }
        }
        if (all_numeric && num >= 0.0f) return num;
    }

    // 2) ISO-style: find date part (YYYY-MM-DD or YYYY/MM/DD) and time part
    int year = 0, month = 0, day = 0, hour = 0, min = 0, sec = 0;
    float frac = 0.0f;
    int n = -1;
    if (len >= 19 && (p[4] == '-' && p[7] == '-' && (p[10] == 'T' || p[10] == ' '))) {
        n = std::sscanf(p, "%d-%d-%d%*c%d:%d:%d", &year, &month, &day, &hour, &min, &sec);
    } else if (len >= 19 && p[4] == '/' && p[7] == '/') {
        n = std::sscanf(p, "%d/%d/%d%*c%d:%d:%d", &year, &month, &day, &hour, &min, &sec);
    } else if (len >= 10 && p[4] == '-' && p[7] == '-') {
        n = std::sscanf(p, "%d-%d-%d", &year, &month, &day);
        hour = min = sec = 0;
    }
    if (n >= 6 || (n >= 3 && year >= 1970 && year <= 2100 && month >= 1 && month <= 12 && day >= 1 && day <= 31)) {
        if (len > 19 && (p[19] == '.' || (len > 18 && p[18] == '.'))) {
            size_t fracStart = (p[19] == '.') ? 19u : 18u;  // point at '.' so strtof gets ".123" -> 0.123
            if (fracStart < len) frac = std::strtof(p + fracStart, &end);
            if (frac < 0.0f || frac >= 1.0f) frac = 0.0f;
        }
        std::tm t = {};
        t.tm_year = year - 1900;
        t.tm_mon = month - 1;
        t.tm_mday = day;
        t.tm_hour = hour;
        t.tm_min = min;
        t.tm_sec = sec;
        t.tm_isdst = -1;
        std::time_t tt = std::mktime(&t);
        if (tt != static_cast<std::time_t>(-1)) return static_cast<float>(tt) + frac;
    }
    return 0.0f;
}
}  // namespace

MongoDBQueryClient::MongoDBQueryClient(const std::string& uri, const std::string& db_name) {
    (void)am_qadf_native::io::get_mongocxx_instance();
    mongocxx::uri mongo_uri(uri);
    client_ = mongocxx::client(mongo_uri);
    db_ = client_[db_name];
}

QueryResult MongoDBQueryClient::queryLaserMonitoringData(
    const std::string& model_id,
    int layer_start,
    int layer_end,
    const std::array<float, 3>& bbox_min,
    const std::array<float, 3>& bbox_max
) {
    QueryResult result;
    result.model_id = model_id;
    result.signal_type = "laser_monitoring_data";
    
    // Build query filter
    bsoncxx::builder::stream::document filter_builder;
    filter_builder << "model_id" << model_id;
    
    // Add layer range filter if specified
    if (layer_start >= 0 && layer_end >= 0 && layer_end >= layer_start) {
        filter_builder << "layer_index" << bsoncxx::builder::stream::open_document
                      << "$gte" << layer_start
                      << "$lte" << layer_end
                      << bsoncxx::builder::stream::close_document;
    }
    
    auto filter = filter_builder.extract();
    
    // Execute query
    auto collection = db_["laser_monitoring_data"];
    auto cursor = collection.find(filter.view());
    
    // Apply bbox filtering during parsing
    const float sentinel_low = std::numeric_limits<float>::lowest();
    const float sentinel_high = std::numeric_limits<float>::max();
    bool use_bbox = (bbox_min[0] != sentinel_low || bbox_min[1] != sentinel_low || bbox_min[2] != sentinel_low ||
                     bbox_max[0] != sentinel_high || bbox_max[1] != sentinel_high || bbox_max[2] != sentinel_high);
    
    // Process results - parse laser-specific fields
    for (auto doc : cursor) {
        try {
            // Extract coordinates first (for bbox filtering)
            std::array<float, 3> point = extractCoordinates(doc);
            
            // Apply bbox filter during parsing
            if (use_bbox) {
                bool in_bbox = true;
                for (int i = 0; i < 3; ++i) {
                    if (point[i] < bbox_min[i] || point[i] > bbox_max[i]) {
                        in_bbox = false;
                        break;
                    }
                }
                if (!in_bbox) {
                    continue; // Skip this document
                }
            }
            
            // Extract layer index (handles multiple field name variations)
            int layer_index = extractLayerIndex(doc);
            
            // Extract process parameters (setpoints/commanded values)
            // Try new field names first, then fall back to old names for backward compatibility
            std::vector<std::string> commanded_power_fields = {
                "commanded_power", "commandedPower", "laser_power", "laserPower", "power", "Power", "P", "p"
            };
            float commanded_power = extractFloatValue(doc, commanded_power_fields, 0.0f);
            
            std::vector<std::string> commanded_speed_fields = {
                "commanded_scan_speed", "commandedScanSpeed", "scan_speed", "scanSpeed", "speed", "Speed", 
                "velocity", "Velocity", "V", "v"
            };
            float commanded_scan_speed = extractFloatValue(doc, commanded_speed_fields, 0.0f);
            
            // Extract energy density (try multiple field name variations)
            std::vector<std::string> energy_fields = {
                "energy_density", "energyDensity", "Energy", "energy", "E", "e"
            };
            float energy_density = extractFloatValue(doc, energy_fields, 0.0f);
            
            // Use commanded values if available, otherwise use old field names
            float laser_power = commanded_power;
            float scan_speed = commanded_scan_speed;
            
            // Extract timestamp (numeric or ISO string from existing DB)
            float timestamp = 0.0f;
            try {
                auto ts_elem = doc["timestamp"];
                if (ts_elem) {
                    if (ts_elem.type() == bsoncxx::type::k_string) {
                        std::string s(ts_elem.get_string().value);
                        timestamp = parseTimestampStringToUnix(s);
                    } else if (ts_elem.type() == bsoncxx::type::k_double) {
                        timestamp = static_cast<float>(ts_elem.get_double().value);
                    } else if (ts_elem.type() == bsoncxx::type::k_int32) {
                        timestamp = static_cast<float>(ts_elem.get_int32().value);
                    } else if (ts_elem.type() == bsoncxx::type::k_int64) {
                        timestamp = static_cast<float>(ts_elem.get_int64().value);
                    }
                }
                if (timestamp == 0.0f) {
                    ts_elem = doc["Timestamp"];
                    if (ts_elem && ts_elem.type() == bsoncxx::type::k_string) {
                        std::string s(ts_elem.get_string().value);
                        timestamp = parseTimestampStringToUnix(s);
                    } else if (ts_elem && ts_elem.type() == bsoncxx::type::k_double) {
                        timestamp = static_cast<float>(ts_elem.get_double().value);
                    }
                }
                if (timestamp == 0.0f) {
                    ts_elem = doc["time"];
                    if (ts_elem && ts_elem.type() == bsoncxx::type::k_string) {
                        std::string s(ts_elem.get_string().value);
                        timestamp = parseTimestampStringToUnix(s);
                    } else if (ts_elem && ts_elem.type() == bsoncxx::type::k_double) {
                        timestamp = static_cast<float>(ts_elem.get_double().value);
                    }
                }
            } catch (...) {
                timestamp = 0.0f;
            }
            
            // Parse temporal sensor data (Category 3)
            QueryResult::LaserTemporalData temporal_data;
            temporal_data.commanded_power = commanded_power;
            temporal_data.commanded_scan_speed = commanded_scan_speed;
            
            // Category 3.1 - Laser Power Sensors (Temporal)
            temporal_data.actual_power = extractFloatValue(doc, {"actual_power", "actualPower", "actual_power"}, 0.0f);
            temporal_data.power_setpoint = extractFloatValue(doc, {"power_setpoint", "powerSetpoint", "power_setpoint"}, commanded_power);
            temporal_data.power_error = extractFloatValue(doc, {"power_error", "powerError", "power_error"}, 0.0f);
            temporal_data.power_stability = extractFloatValue(doc, {"power_stability", "powerStability", "power_stability"}, 0.0f);
            temporal_data.power_fluctuation_amplitude = extractFloatValue(doc, {"power_fluctuation_amplitude", "powerFluctuationAmplitude"}, 0.0f);
            temporal_data.power_fluctuation_frequency = extractFloatValue(doc, {"power_fluctuation_frequency", "powerFluctuationFrequency"}, 0.0f);
            
            // Category 3.2 - Beam Temporal Characteristics
            temporal_data.pulse_frequency = extractFloatValue(doc, {"pulse_frequency", "pulseFrequency"}, 0.0f);
            temporal_data.pulse_duration = extractFloatValue(doc, {"pulse_duration", "pulseDuration"}, 0.0f);
            temporal_data.pulse_energy = extractFloatValue(doc, {"pulse_energy", "pulseEnergy"}, 0.0f);
            temporal_data.duty_cycle = extractFloatValue(doc, {"duty_cycle", "dutyCycle"}, 0.0f);
            temporal_data.beam_modulation_frequency = extractFloatValue(doc, {"beam_modulation_frequency", "beamModulationFrequency"}, 0.0f);
            
            // Category 3.3 - Laser System Health (Temporal)
            temporal_data.laser_temperature = extractFloatValue(doc, {"laser_temperature", "laserTemperature"}, 0.0f);
            temporal_data.laser_cooling_water_temp = extractFloatValue(doc, {"laser_cooling_water_temp", "laserCoolingWaterTemp"}, 0.0f);
            temporal_data.laser_cooling_flow_rate = extractFloatValue(doc, {"laser_cooling_flow_rate", "laserCoolingFlowRate"}, 0.0f);
            temporal_data.laser_power_supply_voltage = extractFloatValue(doc, {"laser_power_supply_voltage", "laserPowerSupplyVoltage"}, 0.0f);
            temporal_data.laser_power_supply_current = extractFloatValue(doc, {"laser_power_supply_current", "laserPowerSupplyCurrent"}, 0.0f);
            temporal_data.laser_diode_current = extractFloatValue(doc, {"laser_diode_current", "laserDiodeCurrent"}, 0.0f);
            temporal_data.laser_diode_temperature = extractFloatValue(doc, {"laser_diode_temperature", "laserDiodeTemperature"}, 0.0f);
            temporal_data.laser_operating_hours = extractFloatValue(doc, {"laser_operating_hours", "laserOperatingHours"}, 0.0f);
            
            // Extract laser_pulse_count as integer
            int laser_pulse_count = 0;
            try {
                auto pulse_count_elem = doc["laser_pulse_count"];
                if (!pulse_count_elem) {
                    pulse_count_elem = doc["laserPulseCount"];
                }
                if (pulse_count_elem) {
                    if (pulse_count_elem.type() == bsoncxx::type::k_int32) {
                        laser_pulse_count = pulse_count_elem.get_int32().value;
                    } else if (pulse_count_elem.type() == bsoncxx::type::k_int64) {
                        laser_pulse_count = static_cast<int>(pulse_count_elem.get_int64().value);
                    } else if (pulse_count_elem.type() == bsoncxx::type::k_double) {
                        laser_pulse_count = static_cast<int>(pulse_count_elem.get_double().value);
                    }
                }
            } catch (...) {
                laser_pulse_count = 0;
            }
            temporal_data.laser_pulse_count = laser_pulse_count;
            
            // Add to result
            result.points.push_back(point);
            result.timestamps.push_back(timestamp);
            result.layers.push_back(layer_index);
            // Temporal sensor data (contains commanded_power, commanded_scan_speed, etc.)
            result.laser_temporal_data.push_back(temporal_data);
        } catch (...) {
            // Skip document if parsing fails
            continue;
        }
    }
    
    return result;
}

QueryResult MongoDBQueryClient::queryISPMThermal(
    const std::string& model_id,
    float time_start,
    float time_end,
    int layer_start,
    int layer_end,
    const std::array<float, 3>& bbox_min,
    const std::array<float, 3>& bbox_max
) {
    QueryResult result;
    result.model_id = model_id;
    result.signal_type = "ispm_thermal";
    
    // Build query filter (same pattern as other ISPM queries - no time filtering)
    bsoncxx::builder::stream::document filter_builder;
    filter_builder << "model_id" << model_id;
    
    // Layer filtering (same pattern as other ISPM queries)
    if (layer_start >= 0 && layer_end >= 0) {
        filter_builder << "layer_index" << bsoncxx::builder::stream::open_document
                      << "$gte" << layer_start
                      << "$lte" << layer_end
                      << bsoncxx::builder::stream::close_document;
    } else if (layer_start >= 0) {
        filter_builder << "layer_index" << bsoncxx::builder::stream::open_document
                      << "$gte" << layer_start
                      << bsoncxx::builder::stream::close_document;
    } else if (layer_end >= 0) {
        filter_builder << "layer_index" << bsoncxx::builder::stream::open_document
                      << "$lte" << layer_end
                      << bsoncxx::builder::stream::close_document;
    }
    
    auto filter = filter_builder.extract();
    
    // Execute query on ispm_thermal_monitoring_data collection (ISPM_Thermal)
    auto collection = db_["ispm_thermal_monitoring_data"];
    auto cursor = collection.find(filter.view());
    
    // Parse results - use specialized ISPM parser to extract all fields in C++
    // Apply bbox filtering during parsing (more efficient than post-processing)
    // Check if bbox filtering is needed (compare against sentinel values)
    const float sentinel_low = std::numeric_limits<float>::lowest();
    const float sentinel_high = std::numeric_limits<float>::max();
    bool use_bbox = (bbox_min[0] != sentinel_low || bbox_min[1] != sentinel_low || bbox_min[2] != sentinel_low ||
                     bbox_max[0] != sentinel_high || bbox_max[1] != sentinel_high || bbox_max[2] != sentinel_high);
    
    for (auto doc : cursor) {
        // Extract coordinates first to check bbox
        std::array<float, 3> coords = extractCoordinates(doc);
        
        // Apply bbox filter if specified
        if (use_bbox) {
            bool in_bbox = true;
            for (int i = 0; i < 3; ++i) {
                if (coords[i] < bbox_min[i] || coords[i] > bbox_max[i]) {
                    in_bbox = false;
                    break;
                }
            }
            if (!in_bbox) {
                continue;  // Skip this document
            }
        }
        
        // Parse document (all calculations happen here in C++)
        parseISPMThermalDocument(doc, result);
    }
    
    return result;
}

QueryResult MongoDBQueryClient::queryISPMOptical(
    const std::string& model_id,
    float time_start,
    float time_end,
    int layer_start,
    int layer_end,
    const std::array<float, 3>& bbox_min,
    const std::array<float, 3>& bbox_max
) {
    QueryResult result;
    result.model_id = model_id;
    result.signal_type = "ispm_optical";
    result.format = "point-based";
    
    // Get collection
    auto collection = db_["ispm_optical_monitoring_data"];
    
    // Build query filter
    bsoncxx::builder::basic::document filter_builder;
    filter_builder.append(bsoncxx::builder::basic::kvp("model_id", model_id));
    
    // Layer filtering
    if (layer_start >= 0 && layer_end >= 0) {
        filter_builder.append(bsoncxx::builder::basic::kvp("layer_index", 
            [layer_start, layer_end](bsoncxx::builder::basic::sub_document sub_doc) {
                sub_doc.append(bsoncxx::builder::basic::kvp("$gte", layer_start));
                sub_doc.append(bsoncxx::builder::basic::kvp("$lte", layer_end));
            }));
    } else if (layer_start >= 0) {
        filter_builder.append(bsoncxx::builder::basic::kvp("layer_index", 
            [layer_start](bsoncxx::builder::basic::sub_document sub_doc) {
                sub_doc.append(bsoncxx::builder::basic::kvp("$gte", layer_start));
            }));
    } else if (layer_end >= 0) {
        filter_builder.append(bsoncxx::builder::basic::kvp("layer_index", 
            [layer_end](bsoncxx::builder::basic::sub_document sub_doc) {
                sub_doc.append(bsoncxx::builder::basic::kvp("$lte", layer_end));
            }));
    }
    
    // Execute query
    auto filter = filter_builder.extract();
    mongocxx::options::find opts;
    opts.batch_size(10000);  // Large batch size for performance
    
    auto cursor = collection.find(filter.view(), opts);
    
    // Parse results - use specialized ISPM optical parser to extract all fields in C++
    // Apply bbox filtering during parsing (more efficient than post-processing)
    const float sentinel_low = std::numeric_limits<float>::lowest();
    const float sentinel_high = std::numeric_limits<float>::max();
    bool use_bbox = (bbox_min[0] != sentinel_low || bbox_min[1] != sentinel_low || bbox_min[2] != sentinel_low ||
                     bbox_max[0] != sentinel_high || bbox_max[1] != sentinel_high || bbox_max[2] != sentinel_high);
    
    for (auto doc : cursor) {
        // Extract coordinates first to check bbox
        std::array<float, 3> coords = extractCoordinates(doc);
        
        // Apply bbox filter if specified
        if (use_bbox) {
            bool in_bbox = true;
            for (int i = 0; i < 3; ++i) {
                if (coords[i] < bbox_min[i] || coords[i] > bbox_max[i]) {
                    in_bbox = false;
                    break;
                }
            }
            if (!in_bbox) {
                continue;  // Skip this document
            }
        }
        
        // Parse document (all calculations happen here in C++)
        parseISPMOpticalDocument(doc, result);
    }
    
    return result;
}

QueryResult MongoDBQueryClient::queryISPMAcoustic(
    const std::string& model_id,
    float time_start,
    float time_end,
    int layer_start,
    int layer_end,
    const std::array<float, 3>& bbox_min,
    const std::array<float, 3>& bbox_max
) {
    QueryResult result;
    result.model_id = model_id;
    result.signal_type = "ispm_acoustic";
    result.format = "point-based";
    
    // Get collection
    auto collection = db_["ispm_acoustic_monitoring_data"];
    
    // Build query filter
    bsoncxx::builder::basic::document filter_builder;
    filter_builder.append(bsoncxx::builder::basic::kvp("model_id", model_id));
    
    // Layer filtering
    if (layer_start >= 0 && layer_end >= 0) {
        filter_builder.append(bsoncxx::builder::basic::kvp("layer_index", 
            [layer_start, layer_end](bsoncxx::builder::basic::sub_document sub_doc) {
                sub_doc.append(bsoncxx::builder::basic::kvp("$gte", layer_start));
                sub_doc.append(bsoncxx::builder::basic::kvp("$lte", layer_end));
            }));
    } else if (layer_start >= 0) {
        filter_builder.append(bsoncxx::builder::basic::kvp("layer_index", 
            [layer_start](bsoncxx::builder::basic::sub_document sub_doc) {
                sub_doc.append(bsoncxx::builder::basic::kvp("$gte", layer_start));
            }));
    } else if (layer_end >= 0) {
        filter_builder.append(bsoncxx::builder::basic::kvp("layer_index", 
            [layer_end](bsoncxx::builder::basic::sub_document sub_doc) {
                sub_doc.append(bsoncxx::builder::basic::kvp("$lte", layer_end));
            }));
    }
    
    // Execute query
    auto filter = filter_builder.extract();
    mongocxx::options::find opts;
    opts.batch_size(10000);  // Large batch size for performance
    
    auto cursor = collection.find(filter.view(), opts);
    
    // Parse results - use specialized ISPM acoustic parser to extract all fields in C++
    // Apply bbox filtering during parsing (more efficient than post-processing)
    const float sentinel_low = std::numeric_limits<float>::lowest();
    const float sentinel_high = std::numeric_limits<float>::max();
    bool use_bbox = (bbox_min[0] != sentinel_low || bbox_min[1] != sentinel_low || bbox_min[2] != sentinel_low ||
                     bbox_max[0] != sentinel_high || bbox_max[1] != sentinel_high || bbox_max[2] != sentinel_high);
    
    for (auto doc : cursor) {
        // Extract coordinates first to check bbox
        std::array<float, 3> coords = extractCoordinates(doc);
        
        // Apply bbox filter if specified
        if (use_bbox) {
            bool in_bbox = true;
            for (int i = 0; i < 3; ++i) {
                if (coords[i] < bbox_min[i] || coords[i] > bbox_max[i]) {
                    in_bbox = false;
                    break;
                }
            }
            if (!in_bbox) {
                continue;  // Skip this document
            }
        }
        
        // Parse document (all extraction happens here in C++)
        parseISPMAcousticDocument(doc, result);
    }
    
    return result;
}

QueryResult MongoDBQueryClient::queryCTScan(const std::string& model_id) {
    QueryResult result;
    result.model_id = model_id;
    result.signal_type = "ct_scan";
    
    // Build query filter
    bsoncxx::builder::stream::document filter_builder;
    filter_builder << "model_id" << model_id;
    
    auto filter = filter_builder.extract();
    
    // Execute query on ct_scans collection (metadata collection)
    auto collection = db_["ct_scans"];
    auto cursor = collection.find(filter.view());
    
    // Parse CT scan metadata
    // CT scan metadata typically contains:
    // - file_path or directory_path (for DICOM series)
    // - format (dicom, nifti, tiff, raw)
    // - dimensions (width, height, depth)
    // - voxel_size
    // - acquisition_date
    // - etc.
    
    for (auto doc : cursor) {
        // Extract file path or directory path
        std::string file_path;
        std::string directory_path;
        std::string format = "dicom";  // Default format
        
        if (doc["file_path"]) {
            file_path = std::string(doc["file_path"].get_string().value);
        }
        if (doc["directory_path"]) {
            directory_path = std::string(doc["directory_path"].get_string().value);
        }
        if (doc["format"]) {
            format = std::string(doc["format"].get_string().value);
        }
        
        // Store path information in result metadata
        // Since QueryResult doesn't have a metadata field for file paths,
        // we can store it in a custom way or extend QueryResult
        // For now, we'll store the first file_path found in a way that can be retrieved
        
        // If we have spatial data (points), parse it
        if (doc["x"] || doc["y"] || doc["z"]) {
            parseDocument(doc, result);
        }
        
        // Store file path information in result for later retrieval
        // Note: This is a limitation - QueryResult doesn't have a metadata field
        // The file path will need to be extracted differently or QueryResult extended
        // For now, we'll use a workaround: store in signal_type or extend later
    }
    
    return result;
}

QueryResult MongoDBQueryClient::queryHatchingData(
    const std::string& model_id,
    int layer_start,
    int layer_end,
    const std::array<float, 3>& bbox_min,
    const std::array<float, 3>& bbox_max
) {
    QueryResult result;
    result.model_id = model_id;
    result.signal_type = "hatching";
    
    // Build query filter
    bsoncxx::builder::stream::document filter_builder;
    filter_builder << "model_id" << model_id;
    
    // Note: Layer filtering will be done in code to handle multiple field name variations
    // Query only by model_id, then filter layers in code
    
    auto filter = filter_builder.extract();
    
    // Execute query on hatching_layers collection
    auto collection = db_["hatching_layers"];
    auto cursor = collection.find(filter.view());
    
    // Helper function to check if point is in bbox
    auto point_in_bbox = [&bbox_min, &bbox_max](const std::array<float, 3>& pt) -> bool {
        if (bbox_min[0] == std::numeric_limits<float>::lowest()) {
            return true;  // No bbox filtering
        }
        return pt[0] >= bbox_min[0] && pt[0] <= bbox_max[0] &&
               pt[1] >= bbox_min[1] && pt[1] <= bbox_max[1] &&
               pt[2] >= bbox_min[2] && pt[2] <= bbox_max[2];
    };
    
    // Process each layer document
    for (auto layer_doc : cursor) {
        // Get layer index (handles multiple field name variations)
        int layer_index = extractLayerIndex(layer_doc);
        
        // Filter by layer range in code (handles any field name)
        if (layer_start >= 0 && layer_end >= 0) {
            if (layer_index < layer_start || layer_index > layer_end) {
                continue;
            }
        }
        
        // Extract contours (if available) - try multiple field name variations
        bsoncxx::array::view contours_array;
        bool has_contours = false;
        if (layer_doc["contours"]) {
            contours_array = layer_doc["contours"].get_array().value;
            has_contours = true;
        } else if (layer_doc["Contours"]) {
            contours_array = layer_doc["Contours"].get_array().value;
            has_contours = true;
        }
        
        if (has_contours) {
            for (auto contour_elem : contours_array) {
                // Convert array element to document view
                if (contour_elem.type() != bsoncxx::type::k_document) {
                    continue;
                }
                auto contour_doc = contour_elem.get_document().value;
                
                // Check for points field variations
                if (!contour_doc["points"] && !contour_doc["Points"]) {
                    continue;
                }
                
                QueryResult::Contour contour;
                
                // Extract sub_type (inner/outer) - try multiple variations
                if (contour_doc["sub_type"]) {
                    contour.sub_type = std::string(contour_doc["sub_type"].get_string().value);
                } else if (contour_doc["subType"]) {
                    contour.sub_type = std::string(contour_doc["subType"].get_string().value);
                } else if (contour_doc["SubType"]) {
                    contour.sub_type = std::string(contour_doc["SubType"].get_string().value);
                } else if (contour_doc["type"]) {
                    contour.sub_type = std::string(contour_doc["type"].get_string().value);
                } else {
                    contour.sub_type = "unknown";
                }
                
                // Extract color - try multiple variations
                if (contour_doc["color"]) {
                    contour.color = std::string(contour_doc["color"].get_string().value);
                } else if (contour_doc["Color"]) {
                    contour.color = std::string(contour_doc["Color"].get_string().value);
                } else {
                    // Default colors based on sub_type (matching visualizer)
                    if (contour.sub_type == "inner") {
                        contour.color = "#f57900";  // Orange for inner
                    } else if (contour.sub_type == "outer") {
                        contour.color = "#204a87";  // Blue for outer
                    } else {
                        contour.color = "k";  // Black for unknown
                    }
                }
                
                // Extract linewidth - try multiple variations
                std::vector<std::string> linewidth_fields = {
                    "linewidth", "lineWidth", "LineWidth", "line_width", "width", "Width"
                };
                contour.linewidth = extractFloatValue(contour_doc, linewidth_fields, 0.0f);
                if (contour.linewidth == 0.0f) {
                    // Default linewidth based on sub_type
                    if (contour.sub_type == "inner") {
                        contour.linewidth = 1.0f;
                    } else if (contour.sub_type == "outer") {
                        contour.linewidth = 1.4f;
                    } else {
                        contour.linewidth = 0.7f;
                    }
                }
                
                // Extract contour points - try multiple field name variations
                bsoncxx::array::view contour_points_array;
                if (contour_doc["points"]) {
                    contour_points_array = contour_doc["points"].get_array().value;
                } else if (contour_doc["Points"]) {
                    contour_points_array = contour_doc["Points"].get_array().value;
                } else {
                    continue;
                }
                for (auto point_elem : contour_points_array) {
                    if (point_elem.type() != bsoncxx::type::k_array) {
                        continue;
                    }
                    
                    auto point_array = point_elem.get_array().value;
                    if (point_array.length() < 2) {
                        continue;
                    }
                    
                    // Extract coordinates (support 2D or 3D points)
                    std::array<float, 3> point = {0.0f, 0.0f, 0.0f};
                    int coord_idx = 0;
                    for (auto coord : point_array) {
                        if (coord_idx < 3) {
                            point[coord_idx] = static_cast<float>(coord.get_double().value);
                        }
                        coord_idx++;
                    }
                    
                    // Apply spatial filtering to contour points
                    if (point_in_bbox(point)) {
                        contour.points.push_back(point);
                    }
                }
                
                // Only add contour if it has points after filtering
                if (!contour.points.empty()) {
                    result.contours.push_back(contour);
                    result.contour_layers.push_back(layer_index);
                }
            }
        }
        
        // Check for vector-based format first (NEW: correct format)
        bool has_vectors = false;
        bsoncxx::array::view vectors_array;
        bsoncxx::array::view vectordata_array;
        
        if (layer_doc["vectors"] && layer_doc["vectordata"]) {
            vectors_array = layer_doc["vectors"].get_array().value;
            vectordata_array = layer_doc["vectordata"].get_array().value;
            has_vectors = true;
            result.format = "vector-based";
        } else if (layer_doc["Vectors"] && layer_doc["Vectordata"]) {
            vectors_array = layer_doc["Vectors"].get_array().value;
            vectordata_array = layer_doc["Vectordata"].get_array().value;
            has_vectors = true;
            result.format = "vector-based";
        }
        
        if (has_vectors) {
            // NEW FORMAT: Vector-based (accurate, preserves structure)
            // Read vectordata first to create lookup map
            std::map<int, QueryResult::VectorData> vectordata_map;
            for (auto vd_elem : vectordata_array) {
                if (vd_elem.type() != bsoncxx::type::k_document) {
                    continue;
                }
                auto vd_doc = vd_elem.get_document().value;
                
                QueryResult::VectorData vd;
                vd.dataindex = extractIntValue(vd_doc, {"dataindex", "dataindex", "dataIndex", "DataIndex"}, -1);
                vd.partid = extractIntValue(vd_doc, {"partid", "partId", "PartId", "part_id"}, 0);
                vd.scanner = extractIntValue(vd_doc, {"scanner", "Scanner"}, 0);
                vd.laserpower = extractFloatValue(vd_doc, {"laserpower", "laserPower", "LaserPower", "power", "Power", "P", "p"}, 0.0f);
                vd.scannerspeed = extractFloatValue(vd_doc, {"scannerspeed", "scannerSpeed", "ScannerSpeed", "speed", "Speed", "velocity", "Velocity", "V", "v"}, 0.0f);
                vd.laser_beam_width = extractFloatValue(vd_doc, {"laser_beam_width", "laserBeamWidth", "beam_width", "beamWidth", "BeamWidth", "width", "Width"}, 0.1f);
                vd.hatch_spacing = extractFloatValue(vd_doc, {"hatch_spacing", "hatchSpacing", "HatchSpacing", "spacing", "Spacing"}, 0.1f);
                vd.layer_index = layer_index;
                
                // Extract type field (case-insensitive)
                if (vd_doc["type"]) {
                    vd.type = std::string(vd_doc["type"].get_string().value);
                } else if (vd_doc["Type"]) {
                    vd.type = std::string(vd_doc["Type"].get_string().value);
                } else {
                    vd.type = "unknown";
                }
                
                if (vd.dataindex >= 0) {
                    vectordata_map[vd.dataindex] = vd;
                    result.vectordata.push_back(vd);
                }
            }
            
            // Read vectors
            for (auto vec_elem : vectors_array) {
                if (vec_elem.type() != bsoncxx::type::k_document) {
                    continue;
                }
                auto vec_doc = vec_elem.get_document().value;
                
                QueryResult::Vector vec;
                vec.x1 = extractFloatValue(vec_doc, {"x1", "X1"}, 0.0f);
                vec.y1 = extractFloatValue(vec_doc, {"y1", "Y1"}, 0.0f);
                vec.x2 = extractFloatValue(vec_doc, {"x2", "X2"}, 0.0f);
                vec.y2 = extractFloatValue(vec_doc, {"y2", "Y2"}, 0.0f);
                vec.z = extractFloatValue(vec_doc, {"z", "Z", "z_position", "zPosition"}, 0.0f);
                vec.timestamp = extractFloatValue(vec_doc, {"timestamp", "Timestamp", "time", "Time"}, 0.0f);
                vec.dataindex = extractIntValue(vec_doc, {"dataindex", "dataindex", "dataIndex", "DataIndex"}, -1);
                
                // Apply spatial filtering if bbox provided
                bool in_bbox = true;
                if (bbox_min[0] != std::numeric_limits<float>::lowest()) {
                    // Check if vector intersects bounding box
                    float mid_x = (vec.x1 + vec.x2) / 2.0f;
                    float mid_y = (vec.y1 + vec.y2) / 2.0f;
                    in_bbox = (
                        (bbox_min[0] <= vec.x1 && vec.x1 <= bbox_max[0] && bbox_min[1] <= vec.y1 && vec.y1 <= bbox_max[1] && bbox_min[2] <= vec.z && vec.z <= bbox_max[2]) ||
                        (bbox_min[0] <= vec.x2 && vec.x2 <= bbox_max[0] && bbox_min[1] <= vec.y2 && vec.y2 <= bbox_max[1] && bbox_min[2] <= vec.z && vec.z <= bbox_max[2]) ||
                        (bbox_min[0] <= mid_x && mid_x <= bbox_max[0] && bbox_min[1] <= mid_y && mid_y <= bbox_max[1] && bbox_min[2] <= vec.z && vec.z <= bbox_max[2])
                    );
                }
                
                if (in_bbox) {
                    result.vectors.push_back(vec);
                    
                    // For backward compatibility, also extract points from vectors
                    result.points.push_back({vec.x1, vec.y1, vec.z});
                    result.points.push_back({vec.x2, vec.y2, vec.z});
                    
                    // Signals come from vectordata (vd.laserpower, vd.scannerspeed, etc.)
                    result.layers.push_back(layer_index);
                    result.layers.push_back(layer_index);
                }
            }
            
            continue;  // Skip old format processing
        }
        
        // OLD FORMAT: Hatches with points (for backward compatibility)
        // Get hatches array - try multiple field name variations
        bsoncxx::array::view hatches_array;
        bool has_hatches = false;
        if (layer_doc["hatches"]) {
            hatches_array = layer_doc["hatches"].get_array().value;
            has_hatches = true;
        } else if (layer_doc["Hatches"]) {
            hatches_array = layer_doc["Hatches"].get_array().value;
            has_hatches = true;
        } else if (layer_doc["infill"]) {
            hatches_array = layer_doc["infill"].get_array().value;
            has_hatches = true;
        }
        
        if (!has_hatches) {
            continue;
        }
        
        result.format = "point-based";  // Legacy format
        
        // Process each hatch (infill pattern)
        for (auto hatch_elem : hatches_array) {
            // Convert array element to document view
            if (hatch_elem.type() != bsoncxx::type::k_document) {
                continue;
            }
            auto hatch_doc = hatch_elem.get_document().value;
            
            if (!hatch_doc["points"] && !hatch_doc["Points"]) {
                continue;
            }
            
            // Get points array - try multiple field name variations
            bsoncxx::array::view points_array;
            if (hatch_doc["points"]) {
                points_array = hatch_doc["points"].get_array().value;
            } else if (hatch_doc["Points"]) {
                points_array = hatch_doc["Points"].get_array().value;
            } else {
                continue;
            }
            if (points_array.length() == 0) {
                continue;
            }
            
            // Extract hatch metadata (infill type, etc.) - try multiple field name variations
            QueryResult::HatchMetadata hatch_meta;
            hatch_meta.hatch_type = "infill";  // Default
            hatch_meta.laser_beam_width = 0.1f;
            hatch_meta.hatch_spacing = 0.1f;
            hatch_meta.overlap_percentage = 0.0f;
            
            // Extract hatch_type
            if (hatch_doc["hatch_type"]) {
                hatch_meta.hatch_type = std::string(hatch_doc["hatch_type"].get_string().value);
            } else if (hatch_doc["hatchType"]) {
                hatch_meta.hatch_type = std::string(hatch_doc["hatchType"].get_string().value);
            } else if (hatch_doc["type"]) {
                hatch_meta.hatch_type = std::string(hatch_doc["type"].get_string().value);
            }
            
            // Extract laser_beam_width
            std::vector<std::string> beam_width_fields = {
                "laser_beam_width", "laserBeamWidth", "beam_width", "beamWidth", 
                "beamWidth", "width", "Width"
            };
            hatch_meta.laser_beam_width = extractFloatValue(hatch_doc, beam_width_fields, 0.1f);
            
            // Extract hatch_spacing
            std::vector<std::string> spacing_fields = {
                "hatch_spacing", "hatchSpacing", "spacing", "Spacing", "hatchSpacing"
            };
            hatch_meta.hatch_spacing = extractFloatValue(hatch_doc, spacing_fields, 0.1f);
            
            // Extract overlap_percentage
            std::vector<std::string> overlap_fields = {
                "overlap_percentage", "overlapPercentage", "overlap", "Overlap", 
                "overlap_percent", "overlapPercent"
            };
            hatch_meta.overlap_percentage = extractFloatValue(hatch_doc, overlap_fields, 0.0f);
            
            // Extract signal values (same for all points in hatch) - use same variations as laser parameters
            std::vector<std::string> power_fields = {
                "laser_power", "laserPower", "power", "Power", "P", "p"
            };
            float laser_power = extractFloatValue(hatch_doc, power_fields, 0.0f);
            
            std::vector<std::string> speed_fields = {
                "scan_speed", "scanSpeed", "speed", "Speed", 
                "velocity", "Velocity", "V", "v"
            };
            float scan_speed = extractFloatValue(hatch_doc, speed_fields, 0.0f);
            
            std::vector<std::string> energy_fields = {
                "energy_density", "energyDensity", "Energy", "energy", "E", "e"
            };
            float energy_density = extractFloatValue(hatch_doc, energy_fields, 0.0f);
            
            // Track starting index for this hatch
            size_t hatch_start_idx = result.points.size();
            
            // Extract points and apply spatial filtering
            for (auto point_elem : points_array) {
                if (point_elem.type() != bsoncxx::type::k_array) {
                    continue;
                }
                
                auto point_array = point_elem.get_array().value;
                if (point_array.length() < 2) {
                    continue;
                }
                
                // Extract coordinates (support 2D or 3D points)
                std::array<float, 3> point = {0.0f, 0.0f, 0.0f};
                int coord_idx = 0;
                for (auto coord : point_array) {
                    if (coord_idx < 3) {
                        point[coord_idx] = static_cast<float>(coord.get_double().value);
                    }
                    coord_idx++;
                }
                
                // Apply spatial filtering
                if (point_in_bbox(point)) {
                    result.points.push_back(point);
                    result.layers.push_back(layer_index);
                    // Signals come from vectordata (already added above)
                }
            }
            
            // Store hatch metadata if this hatch contributed any points
            if (result.points.size() > hatch_start_idx) {
                result.hatch_metadata.push_back(hatch_meta);
                result.hatch_start_indices.push_back(static_cast<int>(hatch_start_idx));
            }
        }
    }
    
    return result;
}

std::string MongoDBQueryClient::getCTScanFilePath(
    const std::string& model_id,
    const std::string& format
) {
    // Build query filter
    bsoncxx::builder::stream::document filter_builder;
    filter_builder << "model_id" << model_id;
    
    auto filter = filter_builder.extract();
    
    // Execute query on ct_scans collection
    auto collection = db_["ct_scans"];
    auto cursor = collection.find(filter.view());
    
    // Get first document (assuming one CT scan per model)
    for (auto doc : cursor) {
        // Extract file path based on format
        if (format == "dicom" || format == "dcm") {
            if (doc["directory_path"]) {
                return std::string(doc["directory_path"].get_string().value);
            } else if (doc["dicom_directory"]) {
                return std::string(doc["dicom_directory"].get_string().value);
            }
        } else if (format == "nifti" || format == "nii") {
            if (doc["file_path"]) {
                return std::string(doc["file_path"].get_string().value);
            } else if (doc["nifti_file"]) {
                return std::string(doc["nifti_file"].get_string().value);
            }
        } else {
            // Try generic file_path
            if (doc["file_path"]) {
                return std::string(doc["file_path"].get_string().value);
            }
        }
        
        // If no specific field found, return empty string
        break;  // Only process first document
    }
    
    return "";  // Not found
}

std::array<float, 3> MongoDBQueryClient::extractCoordinates(const bsoncxx::document::view& doc) {
    std::array<float, 3> point = {0.0f, 0.0f, 0.0f};
    
    // Try spatial_coordinates array first (most common format)
    try {
        auto coords_elem = doc["spatial_coordinates"];
        if (coords_elem && coords_elem.type() == bsoncxx::type::k_array) {
            auto coords_array = coords_elem.get_array().value;
            int idx = 0;
            for (auto coord : coords_array) {
                if (idx < 3) {
                    // Try double first, then int32/int64
                    if (coord.type() == bsoncxx::type::k_double) {
                        point[idx] = static_cast<float>(coord.get_double().value);
                    } else if (coord.type() == bsoncxx::type::k_int32) {
                        point[idx] = static_cast<float>(coord.get_int32().value);
                    } else if (coord.type() == bsoncxx::type::k_int64) {
                        point[idx] = static_cast<float>(coord.get_int64().value);
                    }
                }
                idx++;
            }
            return point;
        }
    } catch (...) {
        // If extraction fails, try individual fields
    }
    
    // Try individual x, y, z fields (case-insensitive variations)
    // Check lowercase first, with proper type checking
    try {
        auto x_elem = doc["x"];
        if (x_elem) {
            if (x_elem.type() == bsoncxx::type::k_double) {
                point[0] = static_cast<float>(x_elem.get_double().value);
            } else if (x_elem.type() == bsoncxx::type::k_int32) {
                point[0] = static_cast<float>(x_elem.get_int32().value);
            } else if (x_elem.type() == bsoncxx::type::k_int64) {
                point[0] = static_cast<float>(x_elem.get_int64().value);
            }
        } else {
            x_elem = doc["X"];
            if (x_elem) {
                if (x_elem.type() == bsoncxx::type::k_double) {
                    point[0] = static_cast<float>(x_elem.get_double().value);
                } else if (x_elem.type() == bsoncxx::type::k_int32) {
                    point[0] = static_cast<float>(x_elem.get_int32().value);
                } else if (x_elem.type() == bsoncxx::type::k_int64) {
                    point[0] = static_cast<float>(x_elem.get_int64().value);
                }
            } else {
                x_elem = doc["x-comp"];
                if (x_elem && x_elem.type() == bsoncxx::type::k_double) {
                    point[0] = static_cast<float>(x_elem.get_double().value);
                } else {
                    x_elem = doc["X-comp"];
                    if (x_elem && x_elem.type() == bsoncxx::type::k_double) {
                        point[0] = static_cast<float>(x_elem.get_double().value);
                    }
                }
            }
        }
    } catch (...) {
        // X coordinate extraction failed, leave as 0.0
    }
    
    try {
        auto y_elem = doc["y"];
        if (y_elem) {
            if (y_elem.type() == bsoncxx::type::k_double) {
                point[1] = static_cast<float>(y_elem.get_double().value);
            } else if (y_elem.type() == bsoncxx::type::k_int32) {
                point[1] = static_cast<float>(y_elem.get_int32().value);
            } else if (y_elem.type() == bsoncxx::type::k_int64) {
                point[1] = static_cast<float>(y_elem.get_int64().value);
            }
        } else {
            y_elem = doc["Y"];
            if (y_elem) {
                if (y_elem.type() == bsoncxx::type::k_double) {
                    point[1] = static_cast<float>(y_elem.get_double().value);
                } else if (y_elem.type() == bsoncxx::type::k_int32) {
                    point[1] = static_cast<float>(y_elem.get_int32().value);
                } else if (y_elem.type() == bsoncxx::type::k_int64) {
                    point[1] = static_cast<float>(y_elem.get_int64().value);
                }
            } else {
                y_elem = doc["y-comp"];
                if (y_elem && y_elem.type() == bsoncxx::type::k_double) {
                    point[1] = static_cast<float>(y_elem.get_double().value);
                } else {
                    y_elem = doc["Y-comp"];
                    if (y_elem && y_elem.type() == bsoncxx::type::k_double) {
                        point[1] = static_cast<float>(y_elem.get_double().value);
                    }
                }
            }
        }
    } catch (...) {
        // Y coordinate extraction failed, leave as 0.0
    }
    
    try {
        auto z_elem = doc["z"];
        if (z_elem) {
            if (z_elem.type() == bsoncxx::type::k_double) {
                point[2] = static_cast<float>(z_elem.get_double().value);
            } else if (z_elem.type() == bsoncxx::type::k_int32) {
                point[2] = static_cast<float>(z_elem.get_int32().value);
            } else if (z_elem.type() == bsoncxx::type::k_int64) {
                point[2] = static_cast<float>(z_elem.get_int64().value);
            }
        } else {
            z_elem = doc["Z"];
            if (z_elem) {
                if (z_elem.type() == bsoncxx::type::k_double) {
                    point[2] = static_cast<float>(z_elem.get_double().value);
                } else if (z_elem.type() == bsoncxx::type::k_int32) {
                    point[2] = static_cast<float>(z_elem.get_int32().value);
                } else if (z_elem.type() == bsoncxx::type::k_int64) {
                    point[2] = static_cast<float>(z_elem.get_int64().value);
                }
            } else {
                z_elem = doc["z-comp"];
                if (z_elem && z_elem.type() == bsoncxx::type::k_double) {
                    point[2] = static_cast<float>(z_elem.get_double().value);
                } else {
                    z_elem = doc["Z-comp"];
                    if (z_elem && z_elem.type() == bsoncxx::type::k_double) {
                        point[2] = static_cast<float>(z_elem.get_double().value);
                    }
                }
            }
        }
    } catch (...) {
        // Z coordinate extraction failed, leave as 0.0
    }
    
    return point;
}

int MongoDBQueryClient::extractLayerIndex(const bsoncxx::document::view& doc) {
    // Try multiple field name variations
    try {
        auto elem = doc["layer_index"];
        if (elem) {
            if (elem.type() == bsoncxx::type::k_int32) {
                return elem.get_int32().value;
            } else if (elem.type() == bsoncxx::type::k_int64) {
                return static_cast<int>(elem.get_int64().value);
            } else if (elem.type() == bsoncxx::type::k_double) {
                return static_cast<int>(elem.get_double().value);
            }
        }
        elem = doc["layer"];
        if (elem) {
            if (elem.type() == bsoncxx::type::k_int32) {
                return elem.get_int32().value;
            } else if (elem.type() == bsoncxx::type::k_int64) {
                return static_cast<int>(elem.get_int64().value);
            } else if (elem.type() == bsoncxx::type::k_double) {
                return static_cast<int>(elem.get_double().value);
            }
        }
        elem = doc["layer_number"];
        if (elem) {
            if (elem.type() == bsoncxx::type::k_int32) {
                return elem.get_int32().value;
            } else if (elem.type() == bsoncxx::type::k_int64) {
                return static_cast<int>(elem.get_int64().value);
            } else if (elem.type() == bsoncxx::type::k_double) {
                return static_cast<int>(elem.get_double().value);
            }
        }
        elem = doc["layerIndex"];
        if (elem) {
            if (elem.type() == bsoncxx::type::k_int32) {
                return elem.get_int32().value;
            } else if (elem.type() == bsoncxx::type::k_int64) {
                return static_cast<int>(elem.get_int64().value);
            } else if (elem.type() == bsoncxx::type::k_double) {
                return static_cast<int>(elem.get_double().value);
            }
        }
        elem = doc["layerNumber"];
        if (elem) {
            if (elem.type() == bsoncxx::type::k_int32) {
                return elem.get_int32().value;
            } else if (elem.type() == bsoncxx::type::k_int64) {
                return static_cast<int>(elem.get_int64().value);
            } else if (elem.type() == bsoncxx::type::k_double) {
                return static_cast<int>(elem.get_double().value);
            }
        }
    } catch (...) {
        // If extraction fails, return default
    }
    return 0;  // Default if not found
}

float MongoDBQueryClient::extractFloatValue(
    const bsoncxx::document::view& doc,
    const std::vector<std::string>& field_names,
    float default_value
) {
    for (const auto& field_name : field_names) {
        try {
            auto elem = doc[field_name.c_str()];
            if (elem) {
                // Try double first (most common)
                if (elem.type() == bsoncxx::type::k_double) {
                    return static_cast<float>(elem.get_double().value);
                }
                // Also try int32/int64 (some databases store numbers as integers)
                else if (elem.type() == bsoncxx::type::k_int32) {
                    return static_cast<float>(elem.get_int32().value);
                }
                else if (elem.type() == bsoncxx::type::k_int64) {
                    return static_cast<float>(elem.get_int64().value);
                }
            }
        } catch (...) {
            // Continue to next field name if extraction fails
            continue;
        }
    }
    return default_value;
}

int MongoDBQueryClient::extractIntValue(
    const bsoncxx::document::view& doc,
    const std::vector<std::string>& field_names,
    int default_value
) {
    for (const auto& field_name : field_names) {
        try {
            auto elem = doc[field_name.c_str()];
            if (elem) {
                // Try int32 first (most common for integers)
                if (elem.type() == bsoncxx::type::k_int32) {
                    return elem.get_int32().value;
                }
                // Also try int64
                else if (elem.type() == bsoncxx::type::k_int64) {
                    return static_cast<int>(elem.get_int64().value);
                }
                // Also try double (some databases store integers as doubles)
                else if (elem.type() == bsoncxx::type::k_double) {
                    return static_cast<int>(elem.get_double().value);
                }
            }
        } catch (...) {
            // Continue to next field name if extraction fails
            continue;
        }
    }
    return default_value;
}

void MongoDBQueryClient::parseDocument(
    const bsoncxx::document::view& doc,
    QueryResult& result
) {
    // Parse MongoDB document
    // Extract: x, y, z, value, timestamp, layer
    
    // Use helper functions for flexible field name handling
    std::array<float, 3> point = extractCoordinates(doc);
    int layer = extractLayerIndex(doc);
    
    float value = 0.0f;
    float timestamp = 0.0f;
    
    // Extract value
    if (doc["value"]) {
        value = static_cast<float>(doc["value"].get_double().value);
    }
    
    // Extract timestamp (try multiple field name variations)
    try {
        if (doc["timestamp"]) {
            if (doc["timestamp"].type() == bsoncxx::type::k_string) {
                std::string s(doc["timestamp"].get_string().value);
                timestamp = parseTimestampStringToUnix(s);
            } else {
                timestamp = static_cast<float>(doc["timestamp"].get_double().value);
            }
        } else if (doc["Timestamp"]) {
            if (doc["Timestamp"].type() == bsoncxx::type::k_string) {
                std::string s(doc["Timestamp"].get_string().value);
                timestamp = parseTimestampStringToUnix(s);
            } else {
                timestamp = static_cast<float>(doc["Timestamp"].get_double().value);
            }
        } else if (doc["TimeStamp"]) {
            if (doc["TimeStamp"].type() == bsoncxx::type::k_string) {
                std::string s(doc["TimeStamp"].get_string().value);
                timestamp = parseTimestampStringToUnix(s);
            } else {
                timestamp = static_cast<float>(doc["TimeStamp"].get_double().value);
            }
        } else if (doc["time"]) {
            if (doc["time"].type() == bsoncxx::type::k_string) {
                std::string s(doc["time"].get_string().value);
                timestamp = parseTimestampStringToUnix(s);
            } else {
                timestamp = static_cast<float>(doc["time"].get_double().value);
            }
        } else if (doc["Time"]) {
            if (doc["Time"].type() == bsoncxx::type::k_string) {
                std::string s(doc["Time"].get_string().value);
                timestamp = parseTimestampStringToUnix(s);
            } else {
                timestamp = static_cast<float>(doc["Time"].get_double().value);
            }
        }
    } catch (...) {
        timestamp = 0.0f;  // Default if extraction fails
    }
    
    // Add to result
    result.points.push_back(point);
    result.values.push_back(value);
    result.timestamps.push_back(timestamp);
    result.layers.push_back(layer);
}

void MongoDBQueryClient::parseISPMThermalDocument(
    const bsoncxx::document::view& doc,
    QueryResult& result
) {
    // Parse ISPM_Thermal-specific MongoDB document
    // Extract: x, y, z, timestamp, layer, and all ISPM_Thermal-specific fields
    
    // Extract basic fields using helper functions
    std::array<float, 3> point = extractCoordinates(doc);
    int layer = extractLayerIndex(doc);
    
    float timestamp = 0.0f;
    try {
        if (doc["timestamp"]) {
            if (doc["timestamp"].type() == bsoncxx::type::k_string) {
                std::string s(doc["timestamp"].get_string().value);
                timestamp = parseTimestampStringToUnix(s);
            } else {
                timestamp = static_cast<float>(doc["timestamp"].get_double().value);
            }
        }
    } catch (...) {
        timestamp = 0.0f;
    }
    
    // Extract ISPM_Thermal-specific fields (all in C++ - no Python calculations)
    QueryResult::ISPMThermalData ispm = {};  // Initialize all fields to zero/default
    
    // Melt pool temperature
    ispm.melt_pool_temperature = extractFloatValue(doc, 
        {"melt_pool_temperature", "meltPoolTemperature", "MeltPoolTemperature", "temperature", "Temperature"}, 
        0.0f);
    
    // Melt pool size (extract from dict: width, length, depth)
    if (doc["melt_pool_size"] && doc["melt_pool_size"].type() == bsoncxx::type::k_document) {
        auto melt_pool_size_doc = doc["melt_pool_size"].get_document().view();
        ispm.melt_pool_width = extractFloatValue(melt_pool_size_doc,
            {"width", "Width", "w", "W"}, 0.0f);
        ispm.melt_pool_length = extractFloatValue(melt_pool_size_doc,
            {"length", "Length", "l", "L"}, 0.0f);
        ispm.melt_pool_depth = extractFloatValue(melt_pool_size_doc,
            {"depth", "Depth", "d", "D"}, 0.0f);
    } else {
        // Try alternative field names
        ispm.melt_pool_width = extractFloatValue(doc,
            {"melt_pool_width", "meltPoolWidth", "MeltPoolWidth"}, 0.0f);
        ispm.melt_pool_length = extractFloatValue(doc,
            {"melt_pool_length", "meltPoolLength", "MeltPoolLength"}, 0.0f);
        ispm.melt_pool_depth = extractFloatValue(doc,
            {"melt_pool_depth", "meltPoolDepth", "MeltPoolDepth"}, 0.0f);
    }
    
    // Peak temperature
    ispm.peak_temperature = extractFloatValue(doc,
        {"peak_temperature", "peakTemperature", "PeakTemperature", "peak_temp", "peakTemp"}, 
        0.0f);
    
    // Cooling rate
    ispm.cooling_rate = extractFloatValue(doc,
        {"cooling_rate", "coolingRate", "CoolingRate", "cooling", "Cooling"}, 
        0.0f);
    
    // Temperature gradient
    ispm.temperature_gradient = extractFloatValue(doc,
        {"temperature_gradient", "temperatureGradient", "TemperatureGradient", 
         "gradient", "Gradient", "temp_gradient", "tempGradient"}, 
        0.0f);
    
    // Additional geometric fields (from research article)
    // Calculate in C++ if not stored in MongoDB (no Python calculations)
    
    // Melt pool area (MPA): Calculate from width and length if not stored
    float stored_area = extractFloatValue(doc,
        {"melt_pool_area", "meltPoolArea", "MeltPoolArea", "area", "Area", "MPA", "mpa"}, 
        0.0f);
    if (stored_area > 0.0f) {
        ispm.melt_pool_area = stored_area;  // Use stored value if available
    } else {
        // Calculate in C++: ellipse area = π * (width/2) * (length/2)
        const float PI = 3.14159265358979323846f;
        ispm.melt_pool_area = PI * (ispm.melt_pool_width / 2.0f) * (ispm.melt_pool_length / 2.0f);
    }
    
    // Melt pool eccentricity (MPE): Calculate from width and length if not stored
    float stored_eccentricity = extractFloatValue(doc,
        {"melt_pool_eccentricity", "meltPoolEccentricity", "MeltPoolEccentricity", 
         "eccentricity", "Eccentricity", "MPE", "mpe"}, 
        0.0f);
    if (stored_eccentricity > 0.0f) {
        ispm.melt_pool_eccentricity = stored_eccentricity;  // Use stored value if available
    } else {
        // Calculate in C++: ratio of width to length (minor/major axis)
        ispm.melt_pool_eccentricity = (ispm.melt_pool_length > 0.0f) 
                                      ? (ispm.melt_pool_width / ispm.melt_pool_length) 
                                      : 0.0f;
    }
    
    // Melt pool perimeter (MPP): Calculate from width and length if not stored
    float stored_perimeter = extractFloatValue(doc,
        {"melt_pool_perimeter", "meltPoolPerimeter", "MeltPoolPerimeter", 
         "perimeter", "Perimeter", "MPP", "mpp"}, 
        0.0f);
    if (stored_perimeter > 0.0f) {
        ispm.melt_pool_perimeter = stored_perimeter;  // Use stored value if available
    } else {
        // Calculate in C++: Ramanujan's ellipse perimeter approximation
        // P ≈ π * [3(a+b) - sqrt((3a+b)(a+3b))] where a=length/2, b=width/2
        const float PI = 3.14159265358979323846f;
        float a = ispm.melt_pool_length / 2.0f;
        float b = ispm.melt_pool_width / 2.0f;
        if (a > 0.0f && b > 0.0f) {
            float sqrt_term = std::sqrt((3.0f * a + b) * (a + 3.0f * b));
            ispm.melt_pool_perimeter = PI * (3.0f * (a + b) - sqrt_term);
        } else {
            ispm.melt_pool_perimeter = 0.0f;
        }
    }
    
    // Time over threshold metrics (TOT) - cooling behavior
    // Calculate in C++ from temperature and cooling_rate if not stored
    
    // TOT1200K - Time above camera sensitivity threshold
    float stored_tot1200 = extractFloatValue(doc,
        {"time_over_threshold_1200K", "timeOverThreshold1200K", "TimeOverThreshold1200K",
         "TOT1200K", "tot1200k", "tot_1200K", "time_over_1200K"}, 
        0.0f);
    if (stored_tot1200 > 0.0f) {
        ispm.time_over_threshold_1200K = stored_tot1200;  // Use stored value if available
    } else {
        // Calculate in C++: t = (T - 1200) / cooling_rate * 1000 (convert to ms)
        float temp_kelvin = ispm.melt_pool_temperature + 273.15f;  // Convert to Kelvin
        if (temp_kelvin > 1200.0f && ispm.cooling_rate > 0.0f) {
            ispm.time_over_threshold_1200K = ((temp_kelvin - 1200.0f) / ispm.cooling_rate) * 1000.0f;
        } else {
            ispm.time_over_threshold_1200K = 0.0f;
        }
    }
    
    // TOT1680K - Time above solidification temperature (~1660K)
    float stored_tot1680 = extractFloatValue(doc,
        {"time_over_threshold_1680K", "timeOverThreshold1680K", "TimeOverThreshold1680K",
         "TOT1680K", "tot1680k", "tot_1680K", "time_over_1680K"}, 
        0.0f);
    if (stored_tot1680 > 0.0f) {
        ispm.time_over_threshold_1680K = stored_tot1680;  // Use stored value if available
    } else {
        // Calculate in C++: t = (T - 1680) / cooling_rate * 1000 (convert to ms)
        float temp_kelvin = ispm.melt_pool_temperature + 273.15f;  // Convert to Kelvin
        if (temp_kelvin > 1680.0f && ispm.cooling_rate > 0.0f) {
            ispm.time_over_threshold_1680K = ((temp_kelvin - 1680.0f) / ispm.cooling_rate) * 1000.0f;
        } else {
            ispm.time_over_threshold_1680K = 0.0f;
        }
    }
    
    // TOT2400K - Time above upper threshold
    float stored_tot2400 = extractFloatValue(doc,
        {"time_over_threshold_2400K", "timeOverThreshold2400K", "TimeOverThreshold2400K",
         "TOT2400K", "tot2400k", "tot_2400K", "time_over_2400K"}, 
        0.0f);
    if (stored_tot2400 > 0.0f) {
        ispm.time_over_threshold_2400K = stored_tot2400;  // Use stored value if available
    } else {
        // Calculate in C++: t = (T - 2400) / cooling_rate * 1000 (convert to ms)
        float temp_kelvin = ispm.melt_pool_temperature + 273.15f;  // Convert to Kelvin
        if (temp_kelvin > 2400.0f && ispm.cooling_rate > 0.0f) {
            ispm.time_over_threshold_2400K = ((temp_kelvin - 2400.0f) / ispm.cooling_rate) * 1000.0f;
        } else {
            ispm.time_over_threshold_2400K = 0.0f;
        }
    }
    
    // Process event (string)
    ispm.process_event = extractStringValue(doc,
        {"process_event", "processEvent", "ProcessEvent", "event", "Event"}, 
        "");
    
    // Add to result
    result.points.push_back(point);
    result.values.push_back(ispm.melt_pool_temperature);  // Use melt_pool_temperature as primary value
    result.timestamps.push_back(timestamp);
    result.layers.push_back(layer);
    result.ispm_thermal_data.push_back(ispm);
}

std::string MongoDBQueryClient::extractStringValue(
    const bsoncxx::document::view& doc,
    const std::vector<std::string>& field_names,
    const std::string& default_value
) {
    for (const auto& field_name : field_names) {
        try {
            auto elem = doc[field_name.c_str()];
            if (elem) {
                // bsoncxx uses k_string for string types
                if (elem.type() == bsoncxx::type::k_string) {
                    return std::string(elem.get_string().value);
                }
            }
        } catch (...) {
            // Continue to next field name
        }
    }
    return default_value;
}

void MongoDBQueryClient::parseISPMOpticalDocument(
    const bsoncxx::document::view& doc,
    QueryResult& result
) {
    // Parse ISPM_Optical-specific MongoDB document
    // Extract: x, y, z, timestamp, layer, and all ISPM_Optical-specific fields
    
    // Extract basic fields using helper functions
    std::array<float, 3> point = extractCoordinates(doc);
    int layer = extractLayerIndex(doc);
    
    float timestamp = 0.0f;
    try {
        if (doc["timestamp"]) {
            if (doc["timestamp"].type() == bsoncxx::type::k_string) {
                std::string s(doc["timestamp"].get_string().value);
                timestamp = parseTimestampStringToUnix(s);
            } else {
                timestamp = static_cast<float>(doc["timestamp"].get_double().value);
            }
        }
    } catch (...) {
        timestamp = 0.0f;
    }
    
    // Extract ISPM_Optical-specific fields (all in C++ - no Python calculations)
    QueryResult::ISPMOpticalData ispm = {};  // Initialize all fields to zero/default
    
    // Photodiode signals
    ispm.photodiode_intensity = extractFloatValue(doc,
        {"photodiode_intensity", "photodiodeIntensity", "PhotodiodeIntensity", "intensity", "Intensity"},
        0.0f);
    ispm.photodiode_frequency = extractFloatValue(doc,
        {"photodiode_frequency", "photodiodeFrequency", "PhotodiodeFrequency", "frequency", "Frequency"},
        0.0f);
    ispm.photodiode_coaxial = extractFloatValue(doc,
        {"photodiode_coaxial", "photodiodeCoaxial", "PhotodiodeCoaxial", "coaxial", "Coaxial"},
        -1.0f);  // -1 indicates not available
    ispm.photodiode_off_axis = extractFloatValue(doc,
        {"photodiode_off_axis", "photodiodeOffAxis", "PhotodiodeOffAxis", "off_axis", "offAxis", "OffAxis"},
        -1.0f);  // -1 indicates not available
    
    // Melt pool brightness/intensity
    ispm.melt_pool_brightness = extractFloatValue(doc,
        {"melt_pool_brightness", "meltPoolBrightness", "MeltPoolBrightness", "brightness", "Brightness"},
        0.0f);
    ispm.melt_pool_intensity_mean = extractFloatValue(doc,
        {"melt_pool_intensity_mean", "meltPoolIntensityMean", "MeltPoolIntensityMean", "intensity_mean", "intensityMean"},
        0.0f);
    ispm.melt_pool_intensity_max = extractFloatValue(doc,
        {"melt_pool_intensity_max", "meltPoolIntensityMax", "MeltPoolIntensityMax", "intensity_max", "intensityMax"},
        0.0f);
    ispm.melt_pool_intensity_std = extractFloatValue(doc,
        {"melt_pool_intensity_std", "meltPoolIntensityStd", "MeltPoolIntensityStd", "intensity_std", "intensityStd"},
        0.0f);
    
    // Spatter detection
    try {
        if (doc["spatter_detected"]) {
            if (doc["spatter_detected"].type() == bsoncxx::type::k_bool) {
                ispm.spatter_detected = doc["spatter_detected"].get_bool().value;
            } else if (doc["spatter_detected"].type() == bsoncxx::type::k_int32) {
                ispm.spatter_detected = (doc["spatter_detected"].get_int32().value != 0);
            } else if (doc["spatter_detected"].type() == bsoncxx::type::k_int64) {
                ispm.spatter_detected = (doc["spatter_detected"].get_int64().value != 0);
            }
        }
    } catch (...) {
        ispm.spatter_detected = false;
    }
    ispm.spatter_intensity = extractFloatValue(doc,
        {"spatter_intensity", "spatterIntensity", "SpatterIntensity"},
        -1.0f);  // -1 indicates not detected
    ispm.spatter_count = extractIntValue(doc,
        {"spatter_count", "spatterCount", "SpatterCount", "spatter_count"},
        0);
    
    // Process stability metrics
    ispm.process_stability = extractFloatValue(doc,
        {"process_stability", "processStability", "ProcessStability", "stability", "Stability"},
        0.0f);
    ispm.intensity_variation = extractFloatValue(doc,
        {"intensity_variation", "intensityVariation", "IntensityVariation", "variation", "Variation"},
        0.0f);
    ispm.signal_to_noise_ratio = extractFloatValue(doc,
        {"signal_to_noise_ratio", "signalToNoiseRatio", "SignalToNoiseRatio", "snr", "SNR", "signal_to_noise"},
        0.0f);
    
    // Melt pool imaging
    try {
        if (doc["melt_pool_image_available"]) {
            if (doc["melt_pool_image_available"].type() == bsoncxx::type::k_bool) {
                ispm.melt_pool_image_available = doc["melt_pool_image_available"].get_bool().value;
            } else if (doc["melt_pool_image_available"].type() == bsoncxx::type::k_int32) {
                ispm.melt_pool_image_available = (doc["melt_pool_image_available"].get_int32().value != 0);
            }
        }
    } catch (...) {
        ispm.melt_pool_image_available = false;
    }
    ispm.melt_pool_area_pixels = extractIntValue(doc,
        {"melt_pool_area_pixels", "meltPoolAreaPixels", "MeltPoolAreaPixels", "area_pixels", "areaPixels"},
        -1);  // -1 indicates not available
    ispm.melt_pool_centroid_x = extractFloatValue(doc,
        {"melt_pool_centroid_x", "meltPoolCentroidX", "MeltPoolCentroidX", "centroid_x", "centroidX"},
        -1.0f);  // -1 indicates not available
    ispm.melt_pool_centroid_y = extractFloatValue(doc,
        {"melt_pool_centroid_y", "meltPoolCentroidY", "MeltPoolCentroidY", "centroid_y", "centroidY"},
        -1.0f);  // -1 indicates not available
    
    // Keyhole detection
    try {
        if (doc["keyhole_detected"]) {
            if (doc["keyhole_detected"].type() == bsoncxx::type::k_bool) {
                ispm.keyhole_detected = doc["keyhole_detected"].get_bool().value;
            } else if (doc["keyhole_detected"].type() == bsoncxx::type::k_int32) {
                ispm.keyhole_detected = (doc["keyhole_detected"].get_int32().value != 0);
            }
        }
    } catch (...) {
        ispm.keyhole_detected = false;
    }
    ispm.keyhole_intensity = extractFloatValue(doc,
        {"keyhole_intensity", "keyholeIntensity", "KeyholeIntensity"},
        -1.0f);  // -1 indicates not detected
    
    // Process events
    ispm.process_event = extractStringValue(doc,
        {"process_event", "processEvent", "ProcessEvent", "event", "Event"},
        "");
    
    // Frequency domain features
    ispm.dominant_frequency = extractFloatValue(doc,
        {"dominant_frequency", "dominantFrequency", "DominantFrequency"},
        -1.0f);  // -1 indicates not available
    ispm.frequency_bandwidth = extractFloatValue(doc,
        {"frequency_bandwidth", "frequencyBandwidth", "FrequencyBandwidth", "bandwidth", "Bandwidth"},
        -1.0f);  // -1 indicates not available
    ispm.spectral_energy = extractFloatValue(doc,
        {"spectral_energy", "spectralEnergy", "SpectralEnergy", "energy", "Energy"},
        -1.0f);  // -1 indicates not available
    
    // Add to result
    result.points.push_back(point);
    result.values.push_back(ispm.photodiode_intensity);  // Use photodiode_intensity as primary value
    result.timestamps.push_back(timestamp);
    result.layers.push_back(layer);
    result.ispm_optical_data.push_back(ispm);
}

void MongoDBQueryClient::parseISPMAcousticDocument(
    const bsoncxx::document::view& doc,
    QueryResult& result
) {
    // Parse ISPM_Acoustic-specific MongoDB document
    // Extract: x, y, z, timestamp, layer, and all ISPM_Acoustic-specific fields
    
    // Extract basic fields using helper functions
    std::array<float, 3> point = extractCoordinates(doc);
    int layer = extractLayerIndex(doc);
    
    float timestamp = 0.0f;
    try {
        if (doc["timestamp"]) {
            if (doc["timestamp"].type() == bsoncxx::type::k_string) {
                std::string s(doc["timestamp"].get_string().value);
                timestamp = parseTimestampStringToUnix(s);
            } else {
                timestamp = static_cast<float>(doc["timestamp"].get_double().value);
            }
        }
    } catch (...) {
        timestamp = 0.0f;
    }
    
    // Extract ISPM_Acoustic-specific fields (all in C++ - no Python calculations)
    QueryResult::ISPMAcousticData ispm = {};  // Initialize all fields to zero/default
    
    // Acoustic emission signals
    ispm.acoustic_amplitude = extractFloatValue(doc,
        {"acoustic_amplitude", "acousticAmplitude", "AcousticAmplitude", "amplitude", "Amplitude"},
        0.0f);
    ispm.acoustic_frequency = extractFloatValue(doc,
        {"acoustic_frequency", "acousticFrequency", "AcousticFrequency", "frequency", "Frequency"},
        0.0f);
    ispm.acoustic_rms = extractFloatValue(doc,
        {"acoustic_rms", "acousticRMS", "AcousticRMS", "rms", "RMS"},
        0.0f);
    ispm.acoustic_peak = extractFloatValue(doc,
        {"acoustic_peak", "acousticPeak", "AcousticPeak", "peak", "Peak"},
        0.0f);
    
    // Frequency domain features
    ispm.dominant_frequency = extractFloatValue(doc,
        {"dominant_frequency", "dominantFrequency", "DominantFrequency"},
        0.0f);
    ispm.frequency_bandwidth = extractFloatValue(doc,
        {"frequency_bandwidth", "frequencyBandwidth", "FrequencyBandwidth", "bandwidth", "Bandwidth"},
        0.0f);
    ispm.spectral_centroid = extractFloatValue(doc,
        {"spectral_centroid", "spectralCentroid", "SpectralCentroid", "centroid", "Centroid"},
        0.0f);
    ispm.spectral_energy = extractFloatValue(doc,
        {"spectral_energy", "spectralEnergy", "SpectralEnergy", "energy", "Energy"},
        0.0f);
    ispm.spectral_rolloff = extractFloatValue(doc,
        {"spectral_rolloff", "spectralRolloff", "SpectralRolloff", "rolloff", "Rolloff"},
        -1.0f);  // -1 indicates not available
    
    // Event detection
    try {
        if (doc["spatter_event_detected"]) {
            if (doc["spatter_event_detected"].type() == bsoncxx::type::k_bool) {
                ispm.spatter_event_detected = doc["spatter_event_detected"].get_bool().value;
            } else if (doc["spatter_event_detected"].type() == bsoncxx::type::k_int32) {
                ispm.spatter_event_detected = (doc["spatter_event_detected"].get_int32().value != 0);
            } else if (doc["spatter_event_detected"].type() == bsoncxx::type::k_int64) {
                ispm.spatter_event_detected = (doc["spatter_event_detected"].get_int64().value != 0);
            }
        }
    } catch (...) {
        ispm.spatter_event_detected = false;
    }
    ispm.spatter_event_amplitude = extractFloatValue(doc,
        {"spatter_event_amplitude", "spatterEventAmplitude", "SpatterEventAmplitude"},
        -1.0f);  // -1 indicates not detected
    
    try {
        if (doc["defect_event_detected"]) {
            if (doc["defect_event_detected"].type() == bsoncxx::type::k_bool) {
                ispm.defect_event_detected = doc["defect_event_detected"].get_bool().value;
            } else if (doc["defect_event_detected"].type() == bsoncxx::type::k_int32) {
                ispm.defect_event_detected = (doc["defect_event_detected"].get_int32().value != 0);
            }
        }
    } catch (...) {
        ispm.defect_event_detected = false;
    }
    ispm.defect_event_amplitude = extractFloatValue(doc,
        {"defect_event_amplitude", "defectEventAmplitude", "DefectEventAmplitude"},
        -1.0f);  // -1 indicates not detected
    
    try {
        if (doc["anomaly_detected"]) {
            if (doc["anomaly_detected"].type() == bsoncxx::type::k_bool) {
                ispm.anomaly_detected = doc["anomaly_detected"].get_bool().value;
            } else if (doc["anomaly_detected"].type() == bsoncxx::type::k_int32) {
                ispm.anomaly_detected = (doc["anomaly_detected"].get_int32().value != 0);
            }
        }
    } catch (...) {
        ispm.anomaly_detected = false;
    }
    ispm.anomaly_type = extractStringValue(doc,
        {"anomaly_type", "anomalyType", "AnomalyType", "anomaly", "Anomaly"},
        "");
    
    // Process stability metrics
    ispm.process_stability = extractFloatValue(doc,
        {"process_stability", "processStability", "ProcessStability", "stability", "Stability"},
        0.0f);
    ispm.acoustic_variation = extractFloatValue(doc,
        {"acoustic_variation", "acousticVariation", "AcousticVariation", "variation", "Variation"},
        0.0f);
    ispm.signal_to_noise_ratio = extractFloatValue(doc,
        {"signal_to_noise_ratio", "signalToNoiseRatio", "SignalToNoiseRatio", "snr", "SNR", "signal_to_noise"},
        0.0f);
    
    // Time-domain features
    ispm.zero_crossing_rate = extractFloatValue(doc,
        {"zero_crossing_rate", "zeroCrossingRate", "ZeroCrossingRate", "zcr", "ZCR"},
        -1.0f);  // -1 indicates not available
    ispm.autocorrelation_peak = extractFloatValue(doc,
        {"autocorrelation_peak", "autocorrelationPeak", "AutocorrelationPeak", "autocorr", "Autocorr"},
        -1.0f);  // -1 indicates not available
    
    // Frequency-domain features
    ispm.harmonic_ratio = extractFloatValue(doc,
        {"harmonic_ratio", "harmonicRatio", "HarmonicRatio", "harmonic", "Harmonic"},
        -1.0f);  // -1 indicates not available
    ispm.spectral_flatness = extractFloatValue(doc,
        {"spectral_flatness", "spectralFlatness", "SpectralFlatness", "flatness", "Flatness"},
        -1.0f);  // -1 indicates not available
    ispm.spectral_crest = extractFloatValue(doc,
        {"spectral_crest", "spectralCrest", "SpectralCrest", "crest", "Crest"},
        -1.0f);  // -1 indicates not available
    
    // Process events
    ispm.process_event = extractStringValue(doc,
        {"process_event", "processEvent", "ProcessEvent", "event", "Event"},
        "");
    
    // Acoustic energy
    ispm.acoustic_energy = extractFloatValue(doc,
        {"acoustic_energy", "acousticEnergy", "AcousticEnergy", "energy", "Energy"},
        0.0f);
    
    // Add to result
    result.points.push_back(point);
    result.values.push_back(ispm.acoustic_amplitude);  // Use acoustic_amplitude as primary value
    result.timestamps.push_back(timestamp);
    result.layers.push_back(layer);
    result.ispm_acoustic_data.push_back(ispm);
}

QueryResult MongoDBQueryClient::queryISPMStrain(
    const std::string& model_id,
    float time_start,
    float time_end,
    int layer_start,
    int layer_end,
    const std::array<float, 3>& bbox_min,
    const std::array<float, 3>& bbox_max
) {
    QueryResult result;
    result.model_id = model_id;
    result.signal_type = "ispm_strain";
    result.format = "point-based";
    
    // Get collection
    auto collection = db_["ispm_strain_monitoring_data"];
    
    // Build query filter
    bsoncxx::builder::basic::document filter_builder;
    filter_builder.append(bsoncxx::builder::basic::kvp("model_id", model_id));
    
    // Layer filtering
    if (layer_start >= 0 && layer_end >= 0) {
        filter_builder.append(bsoncxx::builder::basic::kvp("layer_index", 
            [layer_start, layer_end](bsoncxx::builder::basic::sub_document sub_doc) {
                sub_doc.append(bsoncxx::builder::basic::kvp("$gte", layer_start));
                sub_doc.append(bsoncxx::builder::basic::kvp("$lte", layer_end));
            }));
    } else if (layer_start >= 0) {
        filter_builder.append(bsoncxx::builder::basic::kvp("layer_index", 
            [layer_start](bsoncxx::builder::basic::sub_document sub_doc) {
                sub_doc.append(bsoncxx::builder::basic::kvp("$gte", layer_start));
            }));
    } else if (layer_end >= 0) {
        filter_builder.append(bsoncxx::builder::basic::kvp("layer_index", 
            [layer_end](bsoncxx::builder::basic::sub_document sub_doc) {
                sub_doc.append(bsoncxx::builder::basic::kvp("$lte", layer_end));
            }));
    }
    
    // Execute query
    auto filter = filter_builder.extract();
    mongocxx::options::find opts;
    opts.batch_size(10000);  // Large batch size for performance
    
    auto cursor = collection.find(filter.view(), opts);
    
    // Parse results - use specialized ISPM strain parser to extract all fields in C++
    // Apply bbox filtering during parsing (more efficient than post-processing)
    const float sentinel_low = std::numeric_limits<float>::lowest();
    const float sentinel_high = std::numeric_limits<float>::max();
    bool use_bbox = (bbox_min[0] != sentinel_low || bbox_min[1] != sentinel_low || bbox_min[2] != sentinel_low ||
                     bbox_max[0] != sentinel_high || bbox_max[1] != sentinel_high || bbox_max[2] != sentinel_high);
    
    for (auto doc : cursor) {
        // Extract coordinates first to check bbox
        std::array<float, 3> coords = extractCoordinates(doc);
        
        // Apply bbox filter if specified
        if (use_bbox) {
            bool in_bbox = true;
            for (int i = 0; i < 3; ++i) {
                if (coords[i] < bbox_min[i] || coords[i] > bbox_max[i]) {
                    in_bbox = false;
                    break;
                }
            }
            if (!in_bbox) {
                continue;  // Skip this document
            }
        }
        
        // Parse document (all extraction happens here in C++)
        parseISPMStrainDocument(doc, result);
    }
    
    return result;
}

void MongoDBQueryClient::parseISPMStrainDocument(
    const bsoncxx::document::view& doc,
    QueryResult& result
) {
    // Parse ISPM_Strain-specific MongoDB document
    // Extract: x, y, z, timestamp, layer, and all ISPM_Strain-specific fields
    
    // Extract basic fields using helper functions
    std::array<float, 3> point = extractCoordinates(doc);
    int layer = extractLayerIndex(doc);
    
    float timestamp = 0.0f;
    try {
        if (doc["timestamp"]) {
            if (doc["timestamp"].type() == bsoncxx::type::k_string) {
                std::string s(doc["timestamp"].get_string().value);
                timestamp = parseTimestampStringToUnix(s);
            } else {
                timestamp = static_cast<float>(doc["timestamp"].get_double().value);
            }
        }
    } catch (...) {
        timestamp = 0.0f;
    }
    
    // Extract ISPM_Strain-specific fields (all in C++ - no Python calculations)
    QueryResult::ISPMStrainData ispm = {};  // Initialize all fields to zero/default
    
    // Strain components
    ispm.strain_xx = extractFloatValue(doc,
        {"strain_xx", "strainXX", "StrainXX", "strain_x", "strainX"},
        0.0f);
    ispm.strain_yy = extractFloatValue(doc,
        {"strain_yy", "strainYY", "StrainYY", "strain_y", "strainY"},
        0.0f);
    ispm.strain_zz = extractFloatValue(doc,
        {"strain_zz", "strainZZ", "StrainZZ", "strain_z", "strainZ"},
        0.0f);
    ispm.strain_xy = extractFloatValue(doc,
        {"strain_xy", "strainXY", "StrainXY"},
        0.0f);
    ispm.strain_xz = extractFloatValue(doc,
        {"strain_xz", "strainXZ", "StrainXZ"},
        0.0f);
    ispm.strain_yz = extractFloatValue(doc,
        {"strain_yz", "strainYZ", "StrainYZ"},
        0.0f);
    
    // Principal strains
    ispm.principal_strain_max = extractFloatValue(doc,
        {"principal_strain_max", "principalStrainMax", "PrincipalStrainMax", "max_principal_strain"},
        0.0f);
    ispm.principal_strain_min = extractFloatValue(doc,
        {"principal_strain_min", "principalStrainMin", "PrincipalStrainMin", "min_principal_strain"},
        0.0f);
    ispm.principal_strain_intermediate = extractFloatValue(doc,
        {"principal_strain_intermediate", "principalStrainIntermediate", "PrincipalStrainIntermediate", "intermediate_principal_strain"},
        0.0f);
    
    // Von Mises strain
    ispm.von_mises_strain = extractFloatValue(doc,
        {"von_mises_strain", "vonMisesStrain", "VonMisesStrain", "equivalent_strain", "EquivalentStrain"},
        0.0f);
    
    // Displacement
    ispm.displacement_x = extractFloatValue(doc,
        {"displacement_x", "displacementX", "DisplacementX", "disp_x", "dispX"},
        0.0f);
    ispm.displacement_y = extractFloatValue(doc,
        {"displacement_y", "displacementY", "DisplacementY", "disp_y", "dispY"},
        0.0f);
    ispm.displacement_z = extractFloatValue(doc,
        {"displacement_z", "displacementZ", "DisplacementZ", "disp_z", "dispZ"},
        0.0f);
    ispm.total_displacement = extractFloatValue(doc,
        {"total_displacement", "totalDisplacement", "TotalDisplacement", "total_disp"},
        0.0f);
    
    // Strain rate
    ispm.strain_rate = extractFloatValue(doc,
        {"strain_rate", "strainRate", "StrainRate", "strainRate"},
        0.0f);
    
    // Residual stress
    ispm.residual_stress_xx = extractFloatValue(doc,
        {"residual_stress_xx", "residualStressXX", "ResidualStressXX", "residual_stress_x"},
        -1.0f);  // -1 indicates not available
    ispm.residual_stress_yy = extractFloatValue(doc,
        {"residual_stress_yy", "residualStressYY", "ResidualStressYY", "residual_stress_y"},
        -1.0f);
    ispm.residual_stress_zz = extractFloatValue(doc,
        {"residual_stress_zz", "residualStressZZ", "ResidualStressZZ", "residual_stress_z"},
        -1.0f);
    ispm.von_mises_stress = extractFloatValue(doc,
        {"von_mises_stress", "vonMisesStress", "VonMisesStress", "equivalent_stress", "EquivalentStress"},
        -1.0f);
    
    // Temperature-compensated strain
    ispm.temperature_compensated_strain = extractFloatValue(doc,
        {"temperature_compensated_strain", "temperatureCompensatedStrain", "TemperatureCompensatedStrain", "temp_compensated_strain"},
        -1.0f);  // -1 indicates not available
    
    // Warping/distortion
    try {
        if (doc["warping_detected"]) {
            if (doc["warping_detected"].type() == bsoncxx::type::k_bool) {
                ispm.warping_detected = doc["warping_detected"].get_bool().value;
            } else if (doc["warping_detected"].type() == bsoncxx::type::k_int32) {
                ispm.warping_detected = (doc["warping_detected"].get_int32().value != 0);
            } else if (doc["warping_detected"].type() == bsoncxx::type::k_int64) {
                ispm.warping_detected = (doc["warping_detected"].get_int64().value != 0);
            }
        }
    } catch (...) {
        ispm.warping_detected = false;
    }
    ispm.warping_magnitude = extractFloatValue(doc,
        {"warping_magnitude", "warpingMagnitude", "WarpingMagnitude"},
        -1.0f);  // -1 indicates not detected
    ispm.distortion_angle = extractFloatValue(doc,
        {"distortion_angle", "distortionAngle", "DistortionAngle"},
        -1.0f);  // -1 indicates not detected
    
    // Layer-wise strain accumulation
    ispm.cumulative_strain = extractFloatValue(doc,
        {"cumulative_strain", "cumulativeStrain", "CumulativeStrain"},
        0.0f);
    ispm.layer_strain_increment = extractFloatValue(doc,
        {"layer_strain_increment", "layerStrainIncrement", "LayerStrainIncrement", "layer_strain_inc"},
        0.0f);
    
    // Event detection
    try {
        if (doc["excessive_strain_event"]) {
            if (doc["excessive_strain_event"].type() == bsoncxx::type::k_bool) {
                ispm.excessive_strain_event = doc["excessive_strain_event"].get_bool().value;
            } else if (doc["excessive_strain_event"].type() == bsoncxx::type::k_int32) {
                ispm.excessive_strain_event = (doc["excessive_strain_event"].get_int32().value != 0);
            }
        }
    } catch (...) {
        ispm.excessive_strain_event = false;
    }
    try {
        if (doc["warping_event_detected"]) {
            if (doc["warping_event_detected"].type() == bsoncxx::type::k_bool) {
                ispm.warping_event_detected = doc["warping_event_detected"].get_bool().value;
            } else if (doc["warping_event_detected"].type() == bsoncxx::type::k_int32) {
                ispm.warping_event_detected = (doc["warping_event_detected"].get_int32().value != 0);
            }
        }
    } catch (...) {
        ispm.warping_event_detected = false;
    }
    try {
        if (doc["distortion_event_detected"]) {
            if (doc["distortion_event_detected"].type() == bsoncxx::type::k_bool) {
                ispm.distortion_event_detected = doc["distortion_event_detected"].get_bool().value;
            } else if (doc["distortion_event_detected"].type() == bsoncxx::type::k_int32) {
                ispm.distortion_event_detected = (doc["distortion_event_detected"].get_int32().value != 0);
            }
        }
    } catch (...) {
        ispm.distortion_event_detected = false;
    }
    try {
        if (doc["anomaly_detected"]) {
            if (doc["anomaly_detected"].type() == bsoncxx::type::k_bool) {
                ispm.anomaly_detected = doc["anomaly_detected"].get_bool().value;
            } else if (doc["anomaly_detected"].type() == bsoncxx::type::k_int32) {
                ispm.anomaly_detected = (doc["anomaly_detected"].get_int32().value != 0);
            }
        }
    } catch (...) {
        ispm.anomaly_detected = false;
    }
    ispm.anomaly_type = extractStringValue(doc,
        {"anomaly_type", "anomalyType", "AnomalyType", "anomaly", "Anomaly"},
        "");
    
    // Process stability metrics
    ispm.process_stability = extractFloatValue(doc,
        {"process_stability", "processStability", "ProcessStability", "stability", "Stability"},
        0.0f);
    ispm.strain_variation = extractFloatValue(doc,
        {"strain_variation", "strainVariation", "StrainVariation", "variation", "Variation"},
        0.0f);
    ispm.strain_uniformity = extractFloatValue(doc,
        {"strain_uniformity", "strainUniformity", "StrainUniformity", "uniformity", "Uniformity"},
        0.0f);
    
    // Process events
    ispm.process_event = extractStringValue(doc,
        {"process_event", "processEvent", "ProcessEvent", "event", "Event"},
        "");
    
    // Strain energy
    ispm.strain_energy_density = extractFloatValue(doc,
        {"strain_energy_density", "strainEnergyDensity", "StrainEnergyDensity", "strain_energy"},
        -1.0f);  // -1 indicates not available
    
    // Add to result
    result.points.push_back(point);
    result.values.push_back(ispm.von_mises_strain);  // Use von_mises_strain as primary value
    result.timestamps.push_back(timestamp);
    result.layers.push_back(layer);
    result.ispm_strain_data.push_back(ispm);
}

QueryResult MongoDBQueryClient::queryISPMPlume(
    const std::string& model_id,
    float time_start,
    float time_end,
    int layer_start,
    int layer_end,
    const std::array<float, 3>& bbox_min,
    const std::array<float, 3>& bbox_max
) {
    QueryResult result;
    result.model_id = model_id;
    result.signal_type = "ispm_plume";
    result.format = "point-based";
    
    // Get collection
    auto collection = db_["ispm_plume_monitoring_data"];
    
    // Build query filter
    bsoncxx::builder::basic::document filter_builder;
    filter_builder.append(bsoncxx::builder::basic::kvp("model_id", model_id));
    
    // Layer filtering
    if (layer_start >= 0 && layer_end >= 0) {
        filter_builder.append(bsoncxx::builder::basic::kvp("layer_index", 
            [layer_start, layer_end](bsoncxx::builder::basic::sub_document sub_doc) {
                sub_doc.append(bsoncxx::builder::basic::kvp("$gte", layer_start));
                sub_doc.append(bsoncxx::builder::basic::kvp("$lte", layer_end));
            }));
    } else if (layer_start >= 0) {
        filter_builder.append(bsoncxx::builder::basic::kvp("layer_index", 
            [layer_start](bsoncxx::builder::basic::sub_document sub_doc) {
                sub_doc.append(bsoncxx::builder::basic::kvp("$gte", layer_start));
            }));
    } else if (layer_end >= 0) {
        filter_builder.append(bsoncxx::builder::basic::kvp("layer_index", 
            [layer_end](bsoncxx::builder::basic::sub_document sub_doc) {
                sub_doc.append(bsoncxx::builder::basic::kvp("$lte", layer_end));
            }));
    }
    
    // Execute query
    auto filter = filter_builder.extract();
    mongocxx::options::find opts;
    opts.batch_size(10000);  // Large batch size for performance
    
    auto cursor = collection.find(filter.view(), opts);
    
    // Parse results - use specialized ISPM plume parser to extract all fields in C++
    // Apply bbox filtering during parsing (more efficient than post-processing)
    const float sentinel_low = std::numeric_limits<float>::lowest();
    const float sentinel_high = std::numeric_limits<float>::max();
    bool use_bbox = (bbox_min[0] != sentinel_low || bbox_min[1] != sentinel_low || bbox_min[2] != sentinel_low ||
                     bbox_max[0] != sentinel_high || bbox_max[1] != sentinel_high || bbox_max[2] != sentinel_high);
    
    for (auto doc : cursor) {
        // Extract coordinates first to check bbox
        std::array<float, 3> coords = extractCoordinates(doc);
        
        // Apply bbox filter if specified
        if (use_bbox) {
            bool in_bbox = true;
            for (int i = 0; i < 3; ++i) {
                if (coords[i] < bbox_min[i] || coords[i] > bbox_max[i]) {
                    in_bbox = false;
                    break;
                }
            }
            if (!in_bbox) {
                continue;  // Skip this document
            }
        }
        
        // Parse document (all extraction happens here in C++)
        parseISPMPlumeDocument(doc, result);
    }
    
    return result;
}

void MongoDBQueryClient::parseISPMPlumeDocument(
    const bsoncxx::document::view& doc,
    QueryResult& result
) {
    // Parse ISPM_Plume-specific MongoDB document
    // Extract: x, y, z, timestamp, layer, and all ISPM_Plume-specific fields
    
    // Extract basic fields using helper functions
    std::array<float, 3> point = extractCoordinates(doc);
    int layer = extractLayerIndex(doc);
    
    float timestamp = 0.0f;
    try {
        if (doc["timestamp"]) {
            if (doc["timestamp"].type() == bsoncxx::type::k_string) {
                std::string s(doc["timestamp"].get_string().value);
                timestamp = parseTimestampStringToUnix(s);
            } else {
                timestamp = static_cast<float>(doc["timestamp"].get_double().value);
            }
        }
    } catch (...) {
        timestamp = 0.0f;
    }
    
    // Extract ISPM_Plume-specific fields (all in C++ - no Python calculations)
    QueryResult::ISPMPlumeData ispm = {};  // Initialize all fields to zero/default
    
    // Plume characteristics
    ispm.plume_intensity = extractFloatValue(doc,
        {"plume_intensity", "plumeIntensity", "PlumeIntensity", "intensity", "Intensity"},
        0.0f);
    ispm.plume_density = extractFloatValue(doc,
        {"plume_density", "plumeDensity", "PlumeDensity", "density", "Density"},
        0.0f);
    ispm.plume_temperature = extractFloatValue(doc,
        {"plume_temperature", "plumeTemperature", "PlumeTemperature", "temperature", "Temperature"},
        0.0f);
    ispm.plume_velocity = extractFloatValue(doc,
        {"plume_velocity", "plumeVelocity", "PlumeVelocity", "velocity", "Velocity"},
        0.0f);
    ispm.plume_velocity_x = extractFloatValue(doc,
        {"plume_velocity_x", "plumeVelocityX", "PlumeVelocityX", "velocity_x", "velocityX"},
        0.0f);
    ispm.plume_velocity_y = extractFloatValue(doc,
        {"plume_velocity_y", "plumeVelocityY", "PlumeVelocityY", "velocity_y", "velocityY"},
        0.0f);
    
    // Plume geometry
    ispm.plume_height = extractFloatValue(doc,
        {"plume_height", "plumeHeight", "PlumeHeight", "height", "Height"},
        0.0f);
    ispm.plume_width = extractFloatValue(doc,
        {"plume_width", "plumeWidth", "PlumeWidth", "width", "Width"},
        0.0f);
    ispm.plume_angle = extractFloatValue(doc,
        {"plume_angle", "plumeAngle", "PlumeAngle", "angle", "Angle"},
        0.0f);
    ispm.plume_spread = extractFloatValue(doc,
        {"plume_spread", "plumeSpread", "PlumeSpread", "spread", "Spread"},
        0.0f);
    ispm.plume_area = extractFloatValue(doc,
        {"plume_area", "plumeArea", "PlumeArea", "area", "Area"},
        0.0f);
    
    // Plume composition
    ispm.particle_concentration = extractFloatValue(doc,
        {"particle_concentration", "particleConcentration", "ParticleConcentration", "particle_concentration"},
        0.0f);
    ispm.metal_vapor_concentration = extractFloatValue(doc,
        {"metal_vapor_concentration", "metalVaporConcentration", "MetalVaporConcentration", "metal_vapor"},
        0.0f);
    ispm.gas_composition_ratio = extractFloatValue(doc,
        {"gas_composition_ratio", "gasCompositionRatio", "GasCompositionRatio", "composition_ratio"},
        0.0f);
    
    // Plume dynamics
    ispm.plume_fluctuation_rate = extractFloatValue(doc,
        {"plume_fluctuation_rate", "plumeFluctuationRate", "PlumeFluctuationRate", "fluctuation_rate"},
        0.0f);
    ispm.plume_instability_index = extractFloatValue(doc,
        {"plume_instability_index", "plumeInstabilityIndex", "PlumeInstabilityIndex", "instability_index"},
        0.0f);
    ispm.plume_turbulence = extractFloatValue(doc,
        {"plume_turbulence", "plumeTurbulence", "PlumeTurbulence", "turbulence"},
        0.0f);
    
    // Process quality indicators
    ispm.process_stability = extractFloatValue(doc,
        {"process_stability", "processStability", "ProcessStability", "stability", "Stability"},
        0.0f);
    ispm.plume_stability = extractFloatValue(doc,
        {"plume_stability", "plumeStability", "PlumeStability"},
        0.0f);
    ispm.intensity_variation = extractFloatValue(doc,
        {"intensity_variation", "intensityVariation", "IntensityVariation", "variation", "Variation"},
        0.0f);
    
    // Event detection
    try {
        if (doc["excessive_plume_event"]) {
            if (doc["excessive_plume_event"].type() == bsoncxx::type::k_bool) {
                ispm.excessive_plume_event = doc["excessive_plume_event"].get_bool().value;
            } else if (doc["excessive_plume_event"].type() == bsoncxx::type::k_int32) {
                ispm.excessive_plume_event = (doc["excessive_plume_event"].get_int32().value != 0);
            } else if (doc["excessive_plume_event"].type() == bsoncxx::type::k_int64) {
                ispm.excessive_plume_event = (doc["excessive_plume_event"].get_int64().value != 0);
            }
        }
    } catch (...) {
        ispm.excessive_plume_event = false;
    }
    try {
        if (doc["unstable_plume_event"]) {
            if (doc["unstable_plume_event"].type() == bsoncxx::type::k_bool) {
                ispm.unstable_plume_event = doc["unstable_plume_event"].get_bool().value;
            } else if (doc["unstable_plume_event"].type() == bsoncxx::type::k_int32) {
                ispm.unstable_plume_event = (doc["unstable_plume_event"].get_int32().value != 0);
            }
        }
    } catch (...) {
        ispm.unstable_plume_event = false;
    }
    try {
        if (doc["contamination_event"]) {
            if (doc["contamination_event"].type() == bsoncxx::type::k_bool) {
                ispm.contamination_event = doc["contamination_event"].get_bool().value;
            } else if (doc["contamination_event"].type() == bsoncxx::type::k_int32) {
                ispm.contamination_event = (doc["contamination_event"].get_int32().value != 0);
            }
        }
    } catch (...) {
        ispm.contamination_event = false;
    }
    try {
        if (doc["anomaly_detected"]) {
            if (doc["anomaly_detected"].type() == bsoncxx::type::k_bool) {
                ispm.anomaly_detected = doc["anomaly_detected"].get_bool().value;
            } else if (doc["anomaly_detected"].type() == bsoncxx::type::k_int32) {
                ispm.anomaly_detected = (doc["anomaly_detected"].get_int32().value != 0);
            }
        }
    } catch (...) {
        ispm.anomaly_detected = false;
    }
    ispm.anomaly_type = extractStringValue(doc,
        {"anomaly_type", "anomalyType", "AnomalyType", "anomaly", "Anomaly"},
        "");
    
    // Plume energy metrics
    ispm.plume_energy = extractFloatValue(doc,
        {"plume_energy", "plumeEnergy", "PlumeEnergy", "energy", "Energy"},
        0.0f);
    ispm.energy_density = extractFloatValue(doc,
        {"energy_density", "energyDensity", "EnergyDensity", "energy_density"},
        0.0f);
    
    // Process events
    ispm.process_event = extractStringValue(doc,
        {"process_event", "processEvent", "ProcessEvent", "event", "Event"},
        "");
    
    // Signal-to-noise ratio
    ispm.signal_to_noise_ratio = extractFloatValue(doc,
        {"signal_to_noise_ratio", "signalToNoiseRatio", "SignalToNoiseRatio", "snr", "SNR", "signal_to_noise"},
        0.0f);
    
    // Additional plume features
    ispm.plume_momentum = extractFloatValue(doc,
        {"plume_momentum", "plumeMomentum", "PlumeMomentum", "momentum"},
        -1.0f);  // -1 indicates not available
    ispm.plume_pressure = extractFloatValue(doc,
        {"plume_pressure", "plumePressure", "PlumePressure", "pressure"},
        -1.0f);  // -1 indicates not available
    
    // Add to result
    result.points.push_back(point);
    result.values.push_back(ispm.plume_intensity);  // Use plume_intensity as primary value
    result.timestamps.push_back(timestamp);
    result.layers.push_back(layer);
    result.ispm_plume_data.push_back(ispm);
}

} // namespace query
} // namespace am_qadf_native
