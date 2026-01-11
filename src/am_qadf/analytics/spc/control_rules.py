"""
Control Rules Detection

Detects control rule violations (Western Electric rules, Nelson rules).
Identifies patterns that indicate process instability or special causes.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging

from .control_charts import ControlChartResult

logger = logging.getLogger(__name__)


@dataclass
class ControlRuleViolation:
    """Control rule violation detection."""

    rule_name: str  # 'western_electric_1', 'nelson_1', etc.
    violation_type: str  # 'out_of_control', 'trend', 'pattern', etc.
    affected_points: List[int]  # Indices of violating points
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str  # Human-readable description
    metadata: Dict[str, Any] = field(default_factory=dict)


class ControlRuleDetector:
    """
    Detect control rule violations.

    Implements:
    - Western Electric rules (8 rules)
    - Nelson rules (8 rules, similar but with variations)
    - Custom rule detection
    - Violation severity classification
    """

    def __init__(self):
        """Initialize control rule detector."""
        pass

    def detect_western_electric_rules(self, chart_result: ControlChartResult) -> List[ControlRuleViolation]:
        """
        Detect Western Electric rule violations.

        Western Electric Rules:
        1. One point beyond 3σ limits
        2. Nine consecutive points on same side of center line
        3. Six consecutive points steadily increasing or decreasing
        4. Fourteen consecutive points alternating up and down
        5. Two of three consecutive points beyond 2σ limits (same side)
        6. Four of five consecutive points beyond 1σ limits (same side)
        7. Fifteen consecutive points within 1σ limits (both sides)
        8. Eight consecutive points beyond 1σ limits (both sides)

        Args:
            chart_result: ControlChartResult to analyze

        Returns:
            List of ControlRuleViolation objects
        """
        violations = []

        if len(chart_result.sample_values) == 0:
            return violations

        values = chart_result.sample_values
        cl = chart_result.center_line
        ucl = chart_result.upper_control_limit
        lcl = chart_result.lower_control_limit

        # Calculate sigma zones (assuming 3-sigma control limits)
        sigma = (ucl - cl) / 3.0 if (ucl - cl) > 0 else abs(chart_result.sample_values.std() if len(values) > 0 else 1.0)

        if sigma <= 0:
            logger.warning("Cannot calculate sigma zones, skipping rule detection")
            return violations

        uwl = cl + 2 * sigma  # Upper warning limit (2-sigma)
        lwl = cl - 2 * sigma  # Lower warning limit (2-sigma)
        u1sigma = cl + 1 * sigma  # Upper 1-sigma limit
        l1sigma = cl - 1 * sigma  # Lower 1-sigma limit

        # Rule 1: One point beyond 3σ limits
        rule1_violations = self._rule1_one_point_beyond_3sigma(values, ucl, lcl, cl)
        if rule1_violations:
            violations.append(
                ControlRuleViolation(
                    rule_name="western_electric_1",
                    violation_type="out_of_control",
                    affected_points=rule1_violations,
                    severity="critical",
                    description=f"Rule 1: {len(rule1_violations)} point(s) beyond 3σ control limits",
                )
            )

        # Rule 2: Nine consecutive points on same side of center line
        rule2_violations = self._rule2_nine_points_same_side(values, cl)
        if rule2_violations:
            violations.append(
                ControlRuleViolation(
                    rule_name="western_electric_2",
                    violation_type="pattern",
                    affected_points=rule2_violations,
                    severity="high",
                    description="Rule 2: Nine consecutive points on same side of center line",
                )
            )

        # Rule 3: Six consecutive points steadily increasing or decreasing
        rule3_violations = self._rule3_six_points_trend(values)
        if rule3_violations:
            violations.append(
                ControlRuleViolation(
                    rule_name="western_electric_3",
                    violation_type="trend",
                    affected_points=rule3_violations,
                    severity="high",
                    description="Rule 3: Six consecutive points steadily increasing or decreasing",
                )
            )

        # Rule 4: Fourteen consecutive points alternating up and down
        rule4_violations = self._rule4_fourteen_points_alternating(values)
        if rule4_violations:
            violations.append(
                ControlRuleViolation(
                    rule_name="western_electric_4",
                    violation_type="pattern",
                    affected_points=rule4_violations,
                    severity="medium",
                    description="Rule 4: Fourteen consecutive points alternating up and down",
                )
            )

        # Rule 5: Two of three consecutive points beyond 2σ limits (same side)
        rule5_violations = self._rule5_two_of_three_beyond_2sigma(values, uwl, lwl, cl)
        if rule5_violations:
            violations.append(
                ControlRuleViolation(
                    rule_name="western_electric_5",
                    violation_type="pattern",
                    affected_points=rule5_violations,
                    severity="high",
                    description="Rule 5: Two of three consecutive points beyond 2σ limits (same side)",
                )
            )

        # Rule 6: Four of five consecutive points beyond 1σ limits (same side)
        rule6_violations = self._rule6_four_of_five_beyond_1sigma(values, u1sigma, l1sigma, cl)
        if rule6_violations:
            violations.append(
                ControlRuleViolation(
                    rule_name="western_electric_6",
                    violation_type="pattern",
                    affected_points=rule6_violations,
                    severity="medium",
                    description="Rule 6: Four of five consecutive points beyond 1σ limits (same side)",
                )
            )

        # Rule 7: Fifteen consecutive points within 1σ limits (both sides)
        rule7_violations = self._rule7_fifteen_points_within_1sigma(values, u1sigma, l1sigma)
        if rule7_violations:
            violations.append(
                ControlRuleViolation(
                    rule_name="western_electric_7",
                    violation_type="pattern",
                    affected_points=rule7_violations,
                    severity="low",
                    description="Rule 7: Fifteen consecutive points within 1σ limits (indicates reduced variation)",
                )
            )

        # Rule 8: Eight consecutive points beyond 1σ limits (both sides)
        rule8_violations = self._rule8_eight_points_beyond_1sigma(values, u1sigma, l1sigma)
        if rule8_violations:
            violations.append(
                ControlRuleViolation(
                    rule_name="western_electric_8",
                    violation_type="pattern",
                    affected_points=rule8_violations,
                    severity="medium",
                    description="Rule 8: Eight consecutive points beyond 1σ limits (both sides)",
                )
            )

        return violations

    def detect_nelson_rules(self, chart_result: ControlChartResult) -> List[ControlRuleViolation]:
        """
        Detect Nelson rule violations.

        Nelson Rules are similar to Western Electric but with some variations:
        1. One point beyond 3σ limits (same as WE Rule 1)
        2. Nine consecutive points on same side of center line (same as WE Rule 2)
        3. Six consecutive points steadily increasing or decreasing (same as WE Rule 3)
        4. Fourteen consecutive points alternating (zigzag pattern)
        5. Two of three consecutive points beyond 2σ limits (same as WE Rule 5)
        6. Four of five consecutive points beyond 1σ limits (same as WE Rule 6)
        7. Fifteen consecutive points within 1σ limits (same as WE Rule 7)
        8. Eight consecutive points beyond 1σ limits (same as WE Rule 8)

        Args:
            chart_result: ControlChartResult to analyze

        Returns:
            List of ControlRuleViolation objects
        """
        # Nelson rules are essentially the same as Western Electric
        # The main difference is in interpretation, not detection
        violations = self.detect_western_electric_rules(chart_result)

        # Rename to Nelson rules
        for violation in violations:
            violation.rule_name = violation.rule_name.replace("western_electric", "nelson")

        return violations

    def detect_custom_rules(self, chart_result: ControlChartResult, rules: List[str]) -> List[ControlRuleViolation]:
        """
        Detect custom rule violations.

        Args:
            chart_result: ControlChartResult to analyze
            rules: List of rule names to check

        Returns:
            List of ControlRuleViolation objects
        """
        violations = []

        for rule in rules:
            if rule.startswith("western_electric") or rule.startswith("nelson"):
                # Use existing rule sets
                if "western_electric" in rule:
                    we_violations = self.detect_western_electric_rules(chart_result)
                    violations.extend([v for v in we_violations if v.rule_name == rule])
                elif "nelson" in rule:
                    n_violations = self.detect_nelson_rules(chart_result)
                    violations.extend([v for v in n_violations if v.rule_name == rule])
            else:
                logger.warning(f"Unknown custom rule: {rule}")

        return violations

    def get_rule_severity(self, violation: ControlRuleViolation) -> str:
        """
        Get severity of a rule violation.

        Args:
            violation: ControlRuleViolation object

        Returns:
            Severity string
        """
        return violation.severity

    def prioritize_violations(self, violations: List[ControlRuleViolation]) -> List[ControlRuleViolation]:
        """
        Prioritize violations by severity.

        Args:
            violations: List of ControlRuleViolation objects

        Returns:
            Sorted list of violations (critical -> high -> medium -> low)
        """
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}

        sorted_violations = sorted(violations, key=lambda v: (severity_order.get(v.severity, 99), len(v.affected_points)))

        return sorted_violations

    # Rule implementation helpers
    def _rule1_one_point_beyond_3sigma(self, values: np.ndarray, ucl: float, lcl: float, cl: float) -> List[int]:
        """Rule 1: One point beyond 3σ limits."""
        violations = []
        for i, val in enumerate(values):
            if np.isfinite(val) and (val > ucl or val < lcl):
                violations.append(i)
        return violations

    def _rule2_nine_points_same_side(self, values: np.ndarray, cl: float) -> List[int]:
        """Rule 2: Nine consecutive points on same side of center line."""
        violations = []
        for i in range(len(values) - 8):
            window = values[i : i + 9]
            if all(np.isfinite(window)):
                above = np.sum(window > cl)
                below = np.sum(window < cl)
                if above == 9 or below == 9:
                    violations.extend(range(i, i + 9))
        return list(set(violations))  # Remove duplicates

    def _rule3_six_points_trend(self, values: np.ndarray) -> List[int]:
        """Rule 3: Six consecutive points steadily increasing or decreasing."""
        violations = []
        for i in range(len(values) - 5):
            window = values[i : i + 6]
            if all(np.isfinite(window)):
                # Check for steadily increasing
                increasing = all(window[j] <= window[j + 1] for j in range(5))
                # Check for steadily decreasing
                decreasing = all(window[j] >= window[j + 1] for j in range(5))
                if increasing or decreasing:
                    violations.extend(range(i, i + 6))
        return list(set(violations))

    def _rule4_fourteen_points_alternating(self, values: np.ndarray) -> List[int]:
        """Rule 4: Fourteen consecutive points alternating up and down."""
        violations = []
        for i in range(len(values) - 13):
            window = values[i : i + 14]
            if all(np.isfinite(window)):
                # Check for alternating pattern
                alternating = True
                for j in range(13):
                    diff1 = window[j + 1] - window[j]
                    diff2 = window[j + 2] - window[j + 1] if j < 12 else 0
                    # Alternating: signs should be opposite
                    if j < 12 and diff1 * diff2 >= 0:  # Same sign means not alternating
                        alternating = False
                        break
                if alternating:
                    violations.extend(range(i, i + 14))
        return list(set(violations))

    def _rule5_two_of_three_beyond_2sigma(self, values: np.ndarray, uwl: float, lwl: float, cl: float) -> List[int]:
        """Rule 5: Two of three consecutive points beyond 2σ limits (same side)."""
        violations = []
        for i in range(len(values) - 2):
            window = values[i : i + 3]
            if all(np.isfinite(window)):
                above_upper = np.sum(window > uwl)
                below_lower = np.sum(window < lwl)
                # Two of three beyond limits on same side
                if above_upper >= 2 or below_lower >= 2:
                    violations.extend([i + j for j in range(3) if window[j] > uwl or window[j] < lwl])
        return list(set(violations))

    def _rule6_four_of_five_beyond_1sigma(self, values: np.ndarray, u1sigma: float, l1sigma: float, cl: float) -> List[int]:
        """Rule 6: Four of five consecutive points beyond 1σ limits (same side)."""
        violations = []
        for i in range(len(values) - 4):
            window = values[i : i + 5]
            if all(np.isfinite(window)):
                above_upper = np.sum(window > u1sigma)
                below_lower = np.sum(window < l1sigma)
                # Four of five beyond limits on same side
                if above_upper >= 4 or below_lower >= 4:
                    violations.extend([i + j for j in range(5) if window[j] > u1sigma or window[j] < l1sigma])
        return list(set(violations))

    def _rule7_fifteen_points_within_1sigma(self, values: np.ndarray, u1sigma: float, l1sigma: float) -> List[int]:
        """Rule 7: Fifteen consecutive points within 1σ limits."""
        violations = []
        for i in range(len(values) - 14):
            window = values[i : i + 15]
            if all(np.isfinite(window)):
                # All points within 1σ
                if all((val >= l1sigma) and (val <= u1sigma) for val in window):
                    violations.extend(range(i, i + 15))
        return list(set(violations))

    def _rule8_eight_points_beyond_1sigma(self, values: np.ndarray, u1sigma: float, l1sigma: float) -> List[int]:
        """Rule 8: Eight consecutive points beyond 1σ limits (both sides)."""
        violations = []
        for i in range(len(values) - 7):
            window = values[i : i + 8]
            if all(np.isfinite(window)):
                # All points beyond 1σ (some above, some below)
                beyond = all((val > u1sigma) or (val < l1sigma) for val in window)
                both_sides = np.any(window > u1sigma) and np.any(window < l1sigma)
                if beyond and both_sides:
                    violations.extend(range(i, i + 8))
        return list(set(violations))
