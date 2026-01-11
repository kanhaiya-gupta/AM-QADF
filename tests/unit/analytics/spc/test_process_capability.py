"""
Unit tests for process capability analysis.

Tests for ProcessCapabilityAnalyzer and ProcessCapabilityResult.
"""

import pytest
import numpy as np

from am_qadf.analytics.spc.process_capability import (
    ProcessCapabilityAnalyzer,
    ProcessCapabilityResult,
)
from am_qadf.analytics.spc.spc_client import SPCConfig
from tests.fixtures.spc.capability_data import (
    generate_capable_process_data,
    generate_incapable_process_data,
    generate_shifted_process_data,
    generate_one_sided_spec_data,
)


class TestProcessCapabilityResult:
    """Test suite for ProcessCapabilityResult dataclass."""

    @pytest.mark.unit
    def test_process_capability_result_creation(self):
        """Test creating ProcessCapabilityResult."""
        result = ProcessCapabilityResult(
            cp=1.5,
            cpk=1.3,
            pp=1.4,
            ppk=1.2,
            cpu=1.5,
            cpl=1.3,
            specification_limits=(12.0, 8.0),
            target_value=10.0,
            process_mean=10.1,
            process_std=1.0,
            within_subgroup_std=0.95,
            overall_std=1.05,
            capability_rating="Adequate",
            metadata={"sample_size": 100},
        )

        assert result.cp == 1.5
        assert result.cpk == 1.3
        assert result.pp == 1.4
        assert result.ppk == 1.2
        assert result.cpu == 1.5
        assert result.cpl == 1.3
        assert result.specification_limits == (12.0, 8.0)
        assert result.target_value == 10.0
        assert result.process_mean == 10.1
        assert result.capability_rating == "Adequate"


class TestProcessCapabilityAnalyzer:
    """Test suite for ProcessCapabilityAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create a ProcessCapabilityAnalyzer instance."""
        return ProcessCapabilityAnalyzer()

    @pytest.fixture
    def config(self):
        """Create an SPCConfig instance."""
        return SPCConfig(subgroup_size=5)

    @pytest.mark.unit
    def test_analyzer_creation(self, analyzer):
        """Test creating ProcessCapabilityAnalyzer."""
        assert analyzer is not None

    @pytest.mark.unit
    def test_calculate_capability(self, analyzer, config):
        """Test calculating process capability."""
        data, spec_limits = generate_capable_process_data(n_samples=100, usl=12.0, lsl=8.0, target=10.0, cpk=1.5)

        result = analyzer.calculate_capability(data, spec_limits, target_value=10.0, subgroup_size=5, config=config)

        assert isinstance(result, ProcessCapabilityResult)
        assert result.cp > 0
        assert result.cpk > 0
        assert result.pp > 0
        assert result.ppk > 0
        assert result.cpu > 0
        assert result.cpl > 0
        assert result.specification_limits == spec_limits
        assert result.target_value == 10.0
        assert result.process_mean == pytest.approx(10.0, abs=1.0)

    @pytest.mark.unit
    def test_calculate_capability_invalid_limits(self, analyzer, config):
        """Test calculating capability with invalid specification limits."""
        data = generate_capable_process_data(n_samples=100)[0]
        invalid_limits = (8.0, 12.0)  # USL < LSL (wrong order)

        with pytest.raises(ValueError, match="USL.*must be greater than LSL"):
            analyzer.calculate_capability(data, invalid_limits, config=config)

    @pytest.mark.unit
    def test_calculate_capability_small_sample(self, analyzer, config):
        """Test calculating capability with small sample size."""
        data, spec_limits = generate_capable_process_data(n_samples=20, usl=12.0, lsl=8.0)

        # Warning is logged (logger.warning), not raised (warnings.warn)
        result = analyzer.calculate_capability(data, spec_limits, config=config)

        assert isinstance(result, ProcessCapabilityResult)

    @pytest.mark.unit
    def test_calculate_cp(self, analyzer):
        """Test calculating Cp index."""
        cp = analyzer.calculate_cp(usl=12.0, lsl=8.0, within_std=0.667)

        # Cp = (USL - LSL) / (6 * sigma)
        # Cp = 4.0 / (6 * 0.667) ≈ 1.0
        assert cp == pytest.approx(1.0, abs=0.1)
        assert cp > 0

    @pytest.mark.unit
    def test_calculate_cp_zero_std(self, analyzer):
        """Test calculating Cp with zero standard deviation."""
        cp = analyzer.calculate_cp(usl=12.0, lsl=8.0, within_std=0.0)

        assert cp == 0.0

    @pytest.mark.unit
    def test_calculate_cpk(self, analyzer):
        """Test calculating Cpk index."""
        cpu, cpl, cpk = analyzer.calculate_cpk(usl=12.0, lsl=8.0, mean=10.0, within_std=0.667)

        assert isinstance(cpu, (int, float))
        assert isinstance(cpl, (int, float))
        assert isinstance(cpk, (int, float))
        assert cpk == min(cpu, cpl)
        assert cpu > 0
        assert cpl > 0
        assert cpk > 0

    @pytest.mark.unit
    def test_calculate_cpk_shifted(self, analyzer):
        """Test calculating Cpk for shifted process."""
        # Process shifted towards upper limit
        cpu, cpl, cpk = analyzer.calculate_cpk(usl=12.0, lsl=8.0, mean=11.0, within_std=0.667)

        assert cpu < cpl  # CPU should be smaller (closer to upper limit)
        assert cpk == cpu  # Cpk is min of CPU and CPL

    @pytest.mark.unit
    def test_calculate_cpk_one_sided_usl(self, analyzer):
        """Test calculating Cpk with one-sided specification (USL only)."""
        cpu, cpl, cpk = analyzer.calculate_cpk(usl=12.0, lsl=float("-inf"), mean=10.0, within_std=0.667)

        assert np.isinf(cpl) or cpl == float("inf")
        assert cpu < float("inf")
        assert cpk == cpu

    @pytest.mark.unit
    def test_calculate_pp(self, analyzer):
        """Test calculating Pp index."""
        pp = analyzer.calculate_pp(usl=12.0, lsl=8.0, overall_std=0.667)

        # Pp = (USL - LSL) / (6 * sigma)
        assert pp == pytest.approx(1.0, abs=0.1)
        assert pp > 0

    @pytest.mark.unit
    def test_calculate_ppk(self, analyzer):
        """Test calculating Ppk index."""
        ppu, ppl, ppk = analyzer.calculate_ppk(usl=12.0, lsl=8.0, mean=10.0, overall_std=0.667)

        assert isinstance(ppu, (int, float))
        assert isinstance(ppl, (int, float))
        assert isinstance(ppk, (int, float))
        assert ppk == min(ppu, ppl)
        assert ppu > 0
        assert ppl > 0
        assert ppk > 0

    @pytest.mark.unit
    def test_rate_capability_excellent(self, analyzer):
        """Test rating capability as excellent."""
        rating = analyzer.rate_capability(cpk=1.8)

        assert rating == "Excellent"

    @pytest.mark.unit
    def test_rate_capability_adequate(self, analyzer):
        """Test rating capability as adequate."""
        rating = analyzer.rate_capability(cpk=1.5)

        assert rating == "Adequate"

    @pytest.mark.unit
    def test_rate_capability_marginal(self, analyzer):
        """Test rating capability as marginal."""
        rating = analyzer.rate_capability(cpk=1.2)

        assert rating == "Marginal"

    @pytest.mark.unit
    def test_rate_capability_inadequate(self, analyzer):
        """Test rating capability as inadequate."""
        rating = analyzer.rate_capability(cpk=0.8)

        assert rating == "Inadequate"

    @pytest.mark.unit
    def test_rate_capability_invalid(self, analyzer):
        """Test rating capability with invalid value."""
        rating = analyzer.rate_capability(cpk=np.nan)

        assert rating == "Invalid"

    @pytest.mark.unit
    def test_estimate_ppm(self, analyzer):
        """Test estimating parts per million out of specification."""
        data, spec_limits = generate_capable_process_data(n_samples=100, usl=12.0, lsl=8.0, cpk=1.5)

        # Calculate capability first
        result = analyzer.calculate_capability(data, spec_limits)

        ppm = analyzer.estimate_ppm(
            result.cpk, result.process_mean, spec_limits[0], spec_limits[1], result.process_std  # USL  # LSL
        )

        assert isinstance(ppm, float)
        assert ppm >= 0
        # For capable process (Cpk > 1.5), PPM should be low
        if result.cpk > 1.5:
            assert ppm < 100  # Less than 100 PPM

    @pytest.mark.unit
    def test_compare_short_term_long_term(self, analyzer):
        """Test comparing short-term vs. long-term capability."""
        comparison = analyzer.compare_short_term_long_term(cp=1.5, cpk=1.3, pp=1.2, ppk=1.0)

        assert isinstance(comparison, dict)
        assert "cp" in comparison
        assert "cpk" in comparison
        assert "pp" in comparison
        assert "ppk" in comparison
        assert "capability_gap" in comparison
        assert "centered_gap" in comparison
        assert "stability_assessment" in comparison
        assert "improvement_potential" in comparison
        assert comparison["cp"] > comparison["pp"]  # Short-term usually better
        assert comparison["capability_gap"] > 0

    @pytest.mark.unit
    def test_capability_incapable_process(self, analyzer, config):
        """Test calculating capability for incapable process."""
        data, spec_limits = generate_incapable_process_data(n_samples=100, usl=12.0, lsl=8.0, cpk=0.8)

        result = analyzer.calculate_capability(data, spec_limits, config=config)

        assert result.cpk < 1.0  # Incapable process
        assert result.capability_rating == "Inadequate"

    @pytest.mark.unit
    def test_capability_shifted_process(self, analyzer, config):
        """Test calculating capability for shifted process."""
        data, spec_limits = generate_shifted_process_data(n_samples=100, usl=12.0, lsl=8.0, shift=1.5)

        result = analyzer.calculate_capability(data, spec_limits, target_value=10.0, config=config)

        assert result.process_mean != pytest.approx(10.0, abs=0.5)  # Process is shifted
        assert result.cpk < result.cp  # Cpk should be less than Cp for shifted process

    @pytest.mark.unit
    def test_capability_one_sided_spec(self, analyzer, config):
        """Test calculating capability with one-sided specification."""
        data, spec_limits = generate_one_sided_spec_data(n_samples=100, usl=12.0)

        result = analyzer.calculate_capability(data, spec_limits, target_value=10.0, config=config)

        assert isinstance(result, ProcessCapabilityResult)
        # With one-sided spec, one of CPU or CPL should be inf
        assert np.isinf(result.cpu) or np.isinf(result.cpl)
