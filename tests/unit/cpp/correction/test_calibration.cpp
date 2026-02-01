/**
 * C++ unit tests for Calibration (calibration.cpp).
 * Tests loadFromFile, saveToFile, computeCalibration, validateCalibration.
 *
 * Aligned with docs/Tests/Test_New_Plans.md (tests/unit/cpp/correction/).
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "am_qadf_native/correction/calibration.hpp"
#include <vector>
#include <array>
#include <string>
#include <fstream>
#include <cstdio>

using namespace am_qadf_native::correction;

namespace {

const char* kTestCalFile = "test_calibration_correction.ini";

void writeTestCalFile() {
    std::ofstream f(kTestCalFile);
    f << "[metadata]\n";
    f << "sensor_id=test_sensor\n";
    f << "calibration_type=intrinsic\n\n";
    f << "[parameters]\n";
    f << "fx=100.0\n";
    f << "fy=100.0\n\n";
    f << "[reference_points]\n";
    f << "p0=0,0,0\n";
    f << "p1=1,0,0\n";
    f << "[measured_points]\n";
    f << "p0=0.1,0,0\n";
    f << "p1=1.1,0,0\n";
    f.close();
}

void removeTestCalFile() {
    std::remove(kTestCalFile);
}

} // namespace

TEST_CASE("Calibration: loadFromFile nonexistent returns empty", "[correction][calibration]") {
    Calibration cal;
    CalibrationData data = cal.loadFromFile("nonexistent_cal_file_xyz.ini");
    REQUIRE(data.sensor_id.empty());
    REQUIRE(data.reference_points.empty());
}

TEST_CASE("Calibration: loadFromFile and validate", "[correction][calibration]") {
    writeTestCalFile();
    Calibration cal;
    CalibrationData data = cal.loadFromFile(kTestCalFile);
    removeTestCalFile();
    REQUIRE(data.sensor_id == "test_sensor");
    REQUIRE(data.calibration_type == "intrinsic");
    REQUIRE(data.parameters.count("fx") > 0);
    REQUIRE(data.reference_points.size() == 2);
    REQUIRE(data.measured_points.size() == 2);
}

TEST_CASE("Calibration: saveToFile and load round-trip", "[correction][calibration]") {
    Calibration cal;
    CalibrationData data;
    data.sensor_id = "roundtrip";
    data.calibration_type = "translation";
    data.reference_points = { {0,0,0}, {1,1,1} };
    data.measured_points = { {0.1f,0.1f,0.1f}, {1.1f,1.1f,1.1f} };
    cal.saveToFile(data, kTestCalFile);
    CalibrationData loaded = cal.loadFromFile(kTestCalFile);
    std::remove(kTestCalFile);
    REQUIRE(loaded.sensor_id == data.sensor_id);
    REQUIRE(loaded.calibration_type == data.calibration_type);
}

TEST_CASE("Calibration: computeCalibration empty returns empty data", "[correction][calibration]") {
    Calibration cal;
    std::vector<std::array<float, 3>> ref, meas;
    CalibrationData data = cal.computeCalibration(ref, meas, "translation");
    REQUIRE(data.reference_points.empty());
}

TEST_CASE("Calibration: computeCalibration size mismatch returns data with ref/meas only", "[correction][calibration]") {
    Calibration cal;
    std::vector<std::array<float, 3>> ref = { {0,0,0}, {1,0,0} };
    std::vector<std::array<float, 3>> meas = { {0.1f,0,0} };
    CalibrationData data = cal.computeCalibration(ref, meas, "translation");
    REQUIRE(data.reference_points.size() == 2);
    REQUIRE(data.measured_points.size() == 1);
}

TEST_CASE("Calibration: computeCalibration translation", "[correction][calibration]") {
    Calibration cal;
    std::vector<std::array<float, 3>> ref = { {0,0,0}, {1,0,0}, {0,1,0} };
    std::vector<std::array<float, 3>> meas = { {1,0,0}, {2,0,0}, {1,1,0} };
    CalibrationData data = cal.computeCalibration(ref, meas, "translation");
    REQUIRE(data.calibration_type == "translation");
    REQUIRE(data.parameters.count("tx") > 0);
    REQUIRE(data.parameters.count("ty") > 0);
    REQUIRE(data.parameters.count("tz") > 0);
    REQUIRE_THAT(data.parameters["tx"], Catch::Matchers::WithinAbs(1.0f, 1e-5f));
}

TEST_CASE("Calibration: validateCalibration empty returns false", "[correction][calibration]") {
    Calibration cal;
    CalibrationData data;
    REQUIRE_FALSE(cal.validateCalibration(data));
}

TEST_CASE("Calibration: validateCalibration size mismatch returns false", "[correction][calibration]") {
    Calibration cal;
    CalibrationData data;
    data.reference_points = { {0,0,0} };
    data.measured_points = { {0,0,0}, {1,1,1} };
    REQUIRE_FALSE(cal.validateCalibration(data));
}

TEST_CASE("Calibration: validateCalibration valid returns true", "[correction][calibration]") {
    Calibration cal;
    CalibrationData data;
    data.reference_points = { {0,0,0}, {1,1,1} };
    data.measured_points = { {0.1f,0,0}, {1.1f,1,1} };
    REQUIRE(cal.validateCalibration(data));
}
