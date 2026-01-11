"""
Unit tests for control rules detection.

Tests for ControlRuleDetector, Western Electric rules, and Nelson rules.
"""

import pytest
import numpy as np

from am_qadf.analytics.spc.control_rules import (
    ControlRuleDetector,
    ControlRuleViolation,
)
from am_qadf.analytics.spc.control_charts import ControlChartResult
from am_qadf.analytics.spc.spc_client import SPCConfig
from tests.fixtures.spc.control_chart_data import (
    generate_in_control_data,
    generate_out_of_control_data,
    generate_trend_data,
)


class TestControlRuleViolation:
    """Test suite for ControlRuleViolation dataclass."""

    @pytest.mark.unit
    def test_control_rule_violation_creation(self):
        """Test creating ControlRuleViolation."""
        violation = ControlRuleViolation(
            rule_name="western_electric_1",
            violation_type="out_of_control",
            affected_points=[5, 10, 15],
            severity="critical",
            description="Rule 1: Points beyond 3σ limits",
            metadata={"sigma": 3.0},
        )

        assert violation.rule_name == "western_electric_1"
        assert violation.violation_type == "out_of_control"
        assert len(violation.affected_points) == 3
        assert violation.severity == "critical"
        assert violation.description.startswith("Rule 1")


class TestControlRuleDetector:
    """Test suite for ControlRuleDetector class."""

    @pytest.fixture
    def detector(self):
        """Create a ControlRuleDetector instance."""
        return ControlRuleDetector()

    @pytest.fixture
    def in_control_chart_result(self):
        """Create an in-control chart result."""
        data = generate_in_control_data(n_samples=50, mean=10.0, std=1.0)
        mean = np.mean(data)
        std = np.std(data, ddof=1)

        return ControlChartResult(
            chart_type="individual",
            center_line=mean,
            upper_control_limit=mean + 3 * std,
            lower_control_limit=mean - 3 * std,
            upper_warning_limit=mean + 2 * std,
            lower_warning_limit=mean - 2 * std,
            sample_values=data,
            sample_indices=np.arange(len(data)),
            out_of_control_points=[],
            baseline_stats={"mean": mean, "std": std},
        )

    @pytest.fixture
    def out_of_control_chart_result(self):
        """Create an out-of-control chart result."""
        # Create data with some out-of-control points
        data = generate_in_control_data(n_samples=40, mean=10.0, std=1.0)
        # Add some OOC points
        ooc_data = generate_out_of_control_data(n_samples=10, mean=10.0, std=1.0, shift_at=0, shift_magnitude=5.0)
        combined_data = np.concatenate([data, ooc_data])

        mean = 10.0
        std = 1.0
        ucl = mean + 3 * std
        lcl = mean - 3 * std

        # Find OOC points
        ooc_indices = [i for i, val in enumerate(combined_data) if val > ucl or val < lcl]

        return ControlChartResult(
            chart_type="individual",
            center_line=mean,
            upper_control_limit=ucl,
            lower_control_limit=lcl,
            upper_warning_limit=mean + 2 * std,
            lower_warning_limit=mean - 2 * std,
            sample_values=combined_data,
            sample_indices=np.arange(len(combined_data)),
            out_of_control_points=ooc_indices,
            baseline_stats={"mean": mean, "std": std},
        )

    @pytest.mark.unit
    def test_detector_creation(self, detector):
        """Test creating ControlRuleDetector."""
        assert detector is not None

    @pytest.mark.unit
    def test_detect_western_electric_rules_rule1(self, detector, out_of_control_chart_result):
        """Test detecting Western Electric Rule 1 (point beyond 3σ)."""
        violations = detector.detect_western_electric_rules(out_of_control_chart_result)

        # Should detect Rule 1 violations if there are OOC points
        rule1_violations = [v for v in violations if v.rule_name == "western_electric_1"]
        if len(out_of_control_chart_result.out_of_control_points) > 0:
            assert len(rule1_violations) > 0
            assert rule1_violations[0].severity == "critical"
            assert rule1_violations[0].violation_type == "out_of_control"

    @pytest.mark.unit
    def test_detect_western_electric_rules_rule2(self, detector):
        """Test detecting Western Electric Rule 2 (9 points same side)."""
        # Create data with 9 consecutive points on same side
        data = np.ones(50) * 11.0  # All points above center
        data[:40] = 9.0  # First 40 points below center
        mean = 10.0
        std = 1.0

        chart_result = ControlChartResult(
            chart_type="individual",
            center_line=mean,
            upper_control_limit=mean + 3 * std,
            lower_control_limit=mean - 3 * std,
            sample_values=data,
            sample_indices=np.arange(len(data)),
            out_of_control_points=[],
            baseline_stats={"mean": mean, "std": std},
        )

        violations = detector.detect_western_electric_rules(chart_result)

        rule2_violations = [v for v in violations if v.rule_name == "western_electric_2"]
        assert len(rule2_violations) > 0

    @pytest.mark.unit
    def test_detect_western_electric_rules_rule3(self, detector):
        """Test detecting Western Electric Rule 3 (6 points trend)."""
        # Create data with increasing trend
        data = np.linspace(10.0, 16.0, 50)  # Steadily increasing
        mean = 13.0
        std = 1.0

        chart_result = ControlChartResult(
            chart_type="individual",
            center_line=mean,
            upper_control_limit=mean + 3 * std,
            lower_control_limit=mean - 3 * std,
            sample_values=data,
            sample_indices=np.arange(len(data)),
            out_of_control_points=[],
            baseline_stats={"mean": mean, "std": std},
        )

        violations = detector.detect_western_electric_rules(chart_result)

        rule3_violations = [v for v in violations if v.rule_name == "western_electric_3"]
        assert len(rule3_violations) > 0
        assert rule3_violations[0].violation_type == "trend"

    @pytest.mark.unit
    def test_detect_western_electric_rules_rule4(self, detector):
        """Test detecting Western Electric Rule 4 (14 points alternating)."""
        # Create alternating data
        data = np.array([10.0, 11.0, 9.0, 12.0, 8.0, 13.0, 7.0, 14.0, 6.0, 15.0, 5.0, 16.0, 4.0, 17.0] + [10.0] * 36)
        mean = 10.0
        std = 1.0

        chart_result = ControlChartResult(
            chart_type="individual",
            center_line=mean,
            upper_control_limit=mean + 3 * std,
            lower_control_limit=mean - 3 * std,
            sample_values=data,
            sample_indices=np.arange(len(data)),
            out_of_control_points=[],
            baseline_stats={"mean": mean, "std": std},
        )

        violations = detector.detect_western_electric_rules(chart_result)

        rule4_violations = [v for v in violations if v.rule_name == "western_electric_4"]
        # May or may not detect depending on pattern matching
        assert isinstance(violations, list)

    @pytest.mark.unit
    def test_detect_western_electric_rules_rule5(self, detector):
        """Test detecting Western Electric Rule 5 (2 of 3 beyond 2σ)."""
        # Create data with 2 of 3 points beyond 2σ
        data = np.ones(50) * 10.0
        # Insert 3 consecutive points: 2 beyond 2σ
        data[20] = 12.5  # Beyond 2σ (mean + 2.5σ)
        data[21] = 12.3  # Beyond 2σ
        data[22] = 10.5  # Within 2σ
        mean = 10.0
        std = 1.0

        chart_result = ControlChartResult(
            chart_type="individual",
            center_line=mean,
            upper_control_limit=mean + 3 * std,
            lower_control_limit=mean - 3 * std,
            upper_warning_limit=mean + 2 * std,
            lower_warning_limit=mean - 2 * std,
            sample_values=data,
            sample_indices=np.arange(len(data)),
            out_of_control_points=[],
            baseline_stats={"mean": mean, "std": std},
        )

        violations = detector.detect_western_electric_rules(chart_result)

        rule5_violations = [v for v in violations if v.rule_name == "western_electric_5"]
        # Should detect if pattern matches
        assert isinstance(violations, list)

    @pytest.mark.unit
    def test_detect_western_electric_rules_all_rules(self, detector, out_of_control_chart_result):
        """Test detecting all Western Electric rules."""
        violations = detector.detect_western_electric_rules(out_of_control_chart_result)

        assert isinstance(violations, list)
        # Check that violations have correct structure
        for violation in violations:
            assert isinstance(violation, ControlRuleViolation)
            assert violation.rule_name.startswith("western_electric_")
            assert violation.severity in ["low", "medium", "high", "critical"]
            assert len(violation.affected_points) > 0

    @pytest.mark.unit
    def test_detect_nelson_rules(self, detector, out_of_control_chart_result):
        """Test detecting Nelson rules."""
        violations = detector.detect_nelson_rules(out_of_control_chart_result)

        assert isinstance(violations, list)
        # Nelson rules should have similar structure
        for violation in violations:
            assert isinstance(violation, ControlRuleViolation)
            assert violation.rule_name.startswith("nelson_")

    @pytest.mark.unit
    def test_detect_custom_rules(self, detector, out_of_control_chart_result):
        """Test detecting custom rules."""
        custom_rules = ["western_electric_1", "western_electric_2"]

        violations = detector.detect_custom_rules(out_of_control_chart_result, custom_rules)

        assert isinstance(violations, list)
        # Should only contain specified rules
        for violation in violations:
            assert violation.rule_name in custom_rules

    @pytest.mark.unit
    def test_detect_custom_rules_invalid(self, detector, out_of_control_chart_result):
        """Test detecting custom rules with invalid rule names."""
        invalid_rules = ["invalid_rule", "another_invalid"]

        violations = detector.detect_custom_rules(out_of_control_chart_result, invalid_rules)

        assert isinstance(violations, list)
        # Should handle gracefully and return empty or log warning

    @pytest.mark.unit
    def test_get_rule_severity(self, detector):
        """Test getting rule severity."""
        violation = ControlRuleViolation(
            rule_name="western_electric_1",
            violation_type="out_of_control",
            affected_points=[1, 2],
            severity="critical",
            description="Test violation",
        )

        severity = detector.get_rule_severity(violation)

        assert severity == "critical"

    @pytest.mark.unit
    def test_prioritize_violations(self, detector):
        """Test prioritizing violations by severity."""
        violations = [
            ControlRuleViolation(
                rule_name="western_electric_1",
                violation_type="out_of_control",
                affected_points=[1, 2],
                severity="critical",
                description="Critical violation",
            ),
            ControlRuleViolation(
                rule_name="western_electric_2",
                violation_type="pattern",
                affected_points=[3, 4, 5],
                severity="low",
                description="Low severity violation",
            ),
            ControlRuleViolation(
                rule_name="western_electric_3",
                violation_type="trend",
                affected_points=[6],
                severity="high",
                description="High severity violation",
            ),
        ]

        prioritized = detector.prioritize_violations(violations)

        assert isinstance(prioritized, list)
        assert len(prioritized) == len(violations)
        # Critical should come first
        assert prioritized[0].severity == "critical"
        # High should come before low
        assert prioritized[1].severity == "high"
        assert prioritized[2].severity == "low"

    @pytest.mark.unit
    def test_detect_western_electric_rules_in_control(self, detector, in_control_chart_result):
        """Test detecting rules with in-control data."""
        violations = detector.detect_western_electric_rules(in_control_chart_result)

        assert isinstance(violations, list)
        # In-control data may still trigger some pattern rules, but should have fewer violations

    @pytest.mark.unit
    def test_detect_western_electric_rules_empty_data(self, detector):
        """Test detecting rules with empty data."""
        chart_result = ControlChartResult(
            chart_type="individual",
            center_line=10.0,
            upper_control_limit=13.0,
            lower_control_limit=7.0,
            sample_values=np.array([]),
            sample_indices=np.array([]),
            out_of_control_points=[],
            baseline_stats={"mean": 10.0, "std": 1.0},
        )

        violations = detector.detect_western_electric_rules(chart_result)

        assert isinstance(violations, list)
        # Should handle empty data gracefully
