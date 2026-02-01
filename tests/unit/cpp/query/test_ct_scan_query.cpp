/**
 * C++ unit tests for CTScanQuery (ct_scan_query.cpp).
 * Requires MongoDBQueryClient and CTImageReader; tests skip when MongoDB unavailable.
 * Uses TEST_MONGODB_URL or MONGODB_URL from env (e.g. source development.env) when MongoDB requires auth.
 *
 * Aligned with docs/Tests/Test_New_Plans.md (tests/unit/cpp/query/).
 */

#include <catch2/catch_test_macros.hpp>
#include "am_qadf_native/query/ct_scan_query.hpp"
#include "am_qadf_native/query/ct_image_reader.hpp"
#include "am_qadf_native/query/mongodb_query_client.hpp"
#include <openvdb/openvdb.h>
#include <cstdlib>
#include <memory>
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

TEST_CASE("CTScanQuery: readVolume unknown format returns empty grid", "[query][ct_scan_query]") {
    if (!tryConnectMongoDB()) {
        SKIP("MongoDB not available");
    }
    try {
        openvdb::initialize();
        MongoDBQueryClient client(getMongoURI(), getTestDb());
        CTImageReader reader;
        CTScanQuery query(&client, &reader);
        FloatGridPtr grid = query.readVolume("nonexistent_model_xyz", "unknown_format");
        REQUIRE(grid != nullptr);
        REQUIRE(grid->tree().activeVoxelCount() == 0);
    } catch (const std::exception& e) {
        skipOnAuthError(e);
        throw;
    }
}

TEST_CASE("CTScanQuery: queryAndRead nonexistent model returns empty grid", "[query][ct_scan_query]") {
    if (!tryConnectMongoDB()) {
        SKIP("MongoDB not available");
    }
    try {
        openvdb::initialize();
        MongoDBQueryClient client(getMongoURI(), getTestDb());
        CTImageReader reader;
        CTScanQuery query(&client, &reader);
        FloatGridPtr grid = query.queryAndRead("nonexistent_model_xyz");
        REQUIRE(grid != nullptr);
    } catch (const std::exception& e) {
        skipOnAuthError(e);
        throw;
    }
}

TEST_CASE("CTScanQuery: queryMetadata returns when MongoDB available", "[query][ct_scan_query]") {
    if (!tryConnectMongoDB()) {
        SKIP("MongoDB not available");
    }
    try {
        MongoDBQueryClient client(getMongoURI(), getTestDb());
        CTImageReader reader;
        CTScanQuery query(&client, &reader);
        QueryResult r = query.queryMetadata("nonexistent_model_xyz");
        REQUIRE((r.model_id.empty() || r.model_id == "nonexistent_model_xyz"));
    } catch (const std::exception& e) {
        skipOnAuthError(e);
        throw;
    }
}
