"""
AM-QADF Statistical Process Control (SPC) Module

Statistical Process Control for voxel domain data:
- Control chart generation and analysis (X-bar, R, S, Individual, Moving Range)
- Process capability analysis (Cp, Cpk, Pp, Ppk)
- Multivariate SPC (Hotelling T², PCA-based)
- Control rule detection (Western Electric rules, Nelson rules)
- Baseline calculation and adaptive control limits
"""

from .spc_client import (
    SPCClient,
    SPCConfig,
)

from .control_charts import (
    ControlChartGenerator,
    ControlChartResult,
    XBarChart,
    RChart,
    SChart,
    IndividualChart,
    MovingRangeChart,
)

from .process_capability import (
    ProcessCapabilityAnalyzer,
    ProcessCapabilityResult,
)

from .multivariate_spc import (
    MultivariateSPCAnalyzer,
    MultivariateSPCResult,
)

from .control_rules import (
    ControlRuleDetector,
    ControlRuleViolation,
)

from .baseline_calculation import (
    BaselineCalculator,
    AdaptiveLimitsCalculator,
    BaselineStatistics,
)

from .spc_storage import (
    SPCStorage,
)

__all__ = [
    # Client
    "SPCClient",
    "SPCConfig",
    # Control Charts
    "ControlChartGenerator",
    "ControlChartResult",
    "XBarChart",
    "RChart",
    "SChart",
    "IndividualChart",
    "MovingRangeChart",
    # Process Capability
    "ProcessCapabilityAnalyzer",
    "ProcessCapabilityResult",
    # Multivariate SPC
    "MultivariateSPCAnalyzer",
    "MultivariateSPCResult",
    # Control Rules
    "ControlRuleDetector",
    "ControlRuleViolation",
    # Baseline
    "BaselineCalculator",
    "AdaptiveLimitsCalculator",
    "BaselineStatistics",
    # Storage
    "SPCStorage",
]
