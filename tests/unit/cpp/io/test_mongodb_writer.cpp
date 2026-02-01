/**
 * C++ unit tests for MongoDBWriter (mongodb_writer.cpp).
 * Optional: requires EIGEN_AVAILABLE and mongocxx. Tests skip when MongoDB unavailable.
 *
 * Uses development.env when available: TEST_MONGODB_URL or MONGODB_URL (with db override)
 * so authenticated MongoDB (e.g. mongodb://admin:password@localhost:27017/...) is used.
 *
 * Aligned with docs/Tests/Test_New_Plans.md (tests/unit/cpp/io/).
 */

#ifdef EIGEN_AVAILABLE

#include <catch2/catch_test_macros.hpp>
#include "am_qadf_native/io/mongodb_writer.hpp"
#include <cstdlib>
#include <string>

using namespace am_qadf_native::io;

namespace {

// Default when no env is set (no auth)
const char* kDefaultMongoURI = "mongodb://localhost:27017";
const char* kTestDb = "am_qadf_test";

// URI from environment (e.g. development.env: TEST_MONGODB_URL or MONGODB_URL)
const char* getMongoURI() {
    const char* u = std::getenv("TEST_MONGODB_URL");
    if (u && u[0] != '\0') return u;
    u = std::getenv("MONGODB_URL");
    if (u && u[0] != '\0') return u;
    return kDefaultMongoURI;
}

// Prefer test DB name from env; otherwise use default
const char* getTestDb() {
    const char* d = std::getenv("TEST_MONGODB_DATABASE");
    if (d && d[0] != '\0') return d;
    return kTestDb;
}

bool tryConnectMongoDB() {
    try {
        MongoDBWriter writer(getMongoURI(), getTestDb());
        return true;
    } catch (...) {
        return false;
    }
}

} // namespace

TEST_CASE("MongoDBWriter: invalid URI throws", "[io][mongodb_writer]") {
    try {
        MongoDBWriter writer("invalid://bad-uri", kTestDb);
        REQUIRE((false));  // expected constructor to throw
    } catch (const std::exception&) {
        REQUIRE(true);
    } catch (...) {
        REQUIRE(true);
    }
}

TEST_CASE("MongoDBWriter: deleteProcessedData returns when MongoDB available", "[io][mongodb_writer]") {
    if (!tryConnectMongoDB()) {
        SKIP("MongoDB not available (set TEST_MONGODB_URL or source development.env)");
    }
    try {
        MongoDBWriter writer(getMongoURI(), getTestDb());
        int deleted = writer.deleteProcessedData("nonexistent_model_xyz", "thermal", "");
        REQUIRE(deleted >= 0);
    } catch (const std::exception& e) {
        std::string msg = e.what();
        if (msg.find("Authentication failed") != std::string::npos ||
            msg.find("authentication") != std::string::npos ||
            msg.find("requires auth") != std::string::npos ||
            msg.find("Command delete requires authentication") != std::string::npos) {
            SKIP("MongoDB requires authentication (source development.env or set TEST_MONGODB_URL)");
        }
        throw;
    }
}

#endif // EIGEN_AVAILABLE
