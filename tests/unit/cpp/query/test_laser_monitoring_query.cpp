/**
 * C++ unit tests for LaserMonitoringQuery (laser_monitoring_query.cpp).
 * Requires MongoDBQueryClient; tests skip when MongoDB unavailable.
 * Uses TEST_MONGODB_URL or MONGODB_URL from env (e.g. source development.env) when MongoDB requires auth.
 *
 * Aligned with docs/Tests/Test_New_Plans.md (tests/unit/cpp/query/).
 */

#include <catch2/catch_test_macros.hpp>
#include "am_qadf_native/query/laser_monitoring_query.hpp"
#include "am_qadf_native/query/mongodb_query_client.hpp"
#include <cstdlib>
#include <string>

using namespace am_qadf_native::query;

namespace {

const char* kDefaultMongoURI = "mongodb://localhost:27017";
const char* kTestDb = "am_qadf_test";

const char* getMongoURI() {
    const char* u = std::getenv("TEST_MONGODB_URL");
    if (u && u[0] != '\0') return u;
    u = std::getenv("MONGODB_URL");
    if (u && u[0] != '\0') return u;
    return kDefaultMongoURI;
}

const char* getTestDb() {
    const char* d = std::getenv("TEST_MONGODB_DATABASE");
    if (d && d[0] != '\0') return d;
    return kTestDb;
}

bool tryConnectMongoDB() {
    try {
        MongoDBQueryClient client(getMongoURI(), getTestDb());
        return true;
    } catch (...) {
        return false;
    }
}

void skipOnAuthError(const std::exception& e) {
    std::string msg = e.what();
    if (msg.find("Authentication failed") != std::string::npos ||
        msg.find("authentication") != std::string::npos ||
        msg.find("requires auth") != std::string::npos ||
        msg.find("Command find requires authentication") != std::string::npos) {
        SKIP("MongoDB requires authentication (source development.env or set TEST_MONGODB_URL)");
    }
}

} // namespace

TEST_CASE("LaserMonitoringQuery: queryByLayer returns QueryResult when MongoDB available", "[query][laser_monitoring_query]") {
    if (!tryConnectMongoDB()) {
        SKIP("MongoDB not available");
    }
    try {
        MongoDBQueryClient client(getMongoURI(), getTestDb());
        LaserMonitoringQuery query(&client);
        QueryResult r = query.queryByLayer("nonexistent_model_xyz", 0, 1);
        REQUIRE((r.model_id == "nonexistent_model_xyz" || r.points.empty()));
        REQUIRE(r.signal_type == "laser_monitoring_data");
    } catch (const std::exception& e) {
        skipOnAuthError(e);
        throw;
    }
}

TEST_CASE("LaserMonitoringQuery: queryByTime returns QueryResult when MongoDB available", "[query][laser_monitoring_query]") {
    if (!tryConnectMongoDB()) {
        SKIP("MongoDB not available");
    }
    try {
        MongoDBQueryClient client(getMongoURI(), getTestDb());
        LaserMonitoringQuery query(&client);
        QueryResult r = query.queryByTime("nonexistent_model_xyz", 0.0f, 1.0f);
        REQUIRE(r.signal_type == "laser_monitoring_data");
    } catch (const std::exception& e) {
        skipOnAuthError(e);
        throw;
    }
}

TEST_CASE("LaserMonitoringQuery: queryBySpatialRegion returns QueryResult when MongoDB available", "[query][laser_monitoring_query]") {
    if (!tryConnectMongoDB()) {
        SKIP("MongoDB not available");
    }
    try {
        MongoDBQueryClient client(getMongoURI(), getTestDb());
        LaserMonitoringQuery query(&client);
        float bbox_min[3] = { 0.0f, 0.0f, 0.0f };
        float bbox_max[3] = { 1.0f, 1.0f, 1.0f };
        QueryResult r = query.queryBySpatialRegion("nonexistent_model_xyz", bbox_min, bbox_max);
        REQUIRE(r.signal_type == "laser_monitoring_data");
    } catch (const std::exception& e) {
        skipOnAuthError(e);
        throw;
    }
}
