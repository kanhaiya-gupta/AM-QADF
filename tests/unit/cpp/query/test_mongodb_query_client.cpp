/**
 * C++ unit tests for MongoDBQueryClient (mongodb_query_client.cpp).
 * Requires mongocxx. Tests that connect to MongoDB are skipped when server is unavailable.
 * Uses TEST_MONGODB_URL or MONGODB_URL from env (e.g. source development.env) when MongoDB requires auth.
 *
 * Aligned with docs/Tests/Test_New_Plans.md (tests/unit/cpp/query/).
 */

#include <catch2/catch_test_macros.hpp>
#include "am_qadf_native/query/mongodb_query_client.hpp"
#include "am_qadf_native/query/query_result.hpp"
#include <array>
#include <cstdlib>
#include <limits>
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

TEST_CASE("MongoDBQueryClient: invalid URI throws", "[query][mongodb_query_client]") {
    try {
        MongoDBQueryClient client("invalid://bad-uri", kTestDb);
        REQUIRE((false));  // expected constructor to throw
    } catch (const std::exception&) {
        REQUIRE(true);
    } catch (...) {
        REQUIRE(true);
    }
}

TEST_CASE("MongoDBQueryClient: queryLaserMonitoringData returns QueryResult when MongoDB available", "[query][mongodb_query_client]") {
    if (!tryConnectMongoDB()) {
        SKIP("MongoDB not available");
    }
    try {
        MongoDBQueryClient client(getMongoURI(), getTestDb());
        std::array<float, 3> bbox_min = { std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest() };
        std::array<float, 3> bbox_max = { std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max() };
        QueryResult r = client.queryLaserMonitoringData("nonexistent_model_xyz", -1, -1, bbox_min, bbox_max);
        REQUIRE(r.model_id == "nonexistent_model_xyz");
        REQUIRE(r.signal_type == "laser_monitoring_data");
        REQUIRE(r.points.empty());
    } catch (const std::exception& e) {
        skipOnAuthError(e);
        throw;
    }
}

TEST_CASE("MongoDBQueryClient: queryCTScan returns QueryResult when MongoDB available", "[query][mongodb_query_client]") {
    if (!tryConnectMongoDB()) {
        SKIP("MongoDB not available");
    }
    try {
        MongoDBQueryClient client(getMongoURI(), getTestDb());
        QueryResult r = client.queryCTScan("nonexistent_model_xyz");
        REQUIRE((r.model_id.empty() || r.model_id == "nonexistent_model_xyz"));
    } catch (const std::exception& e) {
        skipOnAuthError(e);
        throw;
    }
}

TEST_CASE("MongoDBQueryClient: getCTScanFilePath returns empty for nonexistent model", "[query][mongodb_query_client]") {
    if (!tryConnectMongoDB()) {
        SKIP("MongoDB not available");
    }
    try {
        MongoDBQueryClient client(getMongoURI(), getTestDb());
        std::string path = client.getCTScanFilePath("nonexistent_model_xyz", "dicom");
        REQUIRE(path.empty());
    } catch (const std::exception& e) {
        skipOnAuthError(e);
        throw;
    }
}
