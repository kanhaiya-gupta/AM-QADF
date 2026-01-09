"""
Unit tests for signal generation.

Tests for ThermalFieldGenerator, DensityFieldEstimator, and StressFieldGenerator.
"""

import pytest
import numpy as np
from am_qadf.processing.signal_generation import (
    ThermalFieldGenerator,
    DensityFieldEstimator,
    StressFieldGenerator,
)


class TestThermalFieldGenerator:
    """Test suite for ThermalFieldGenerator class."""

    @pytest.fixture
    def generator(self):
        """Create a ThermalFieldGenerator instance."""
        return ThermalFieldGenerator()

    @pytest.mark.unit
    def test_thermal_field_generator_creation_default(self):
        """Test creating ThermalFieldGenerator with default parameters."""
        generator = ThermalFieldGenerator()

        assert generator.thermal_diffusivity == 8.0e-6
        assert generator.heat_capacity == 500.0
        assert generator.density == 8000.0
        assert generator.ambient_temp == 25.0
        assert generator.cooling_rate == 0.1

    @pytest.mark.unit
    def test_thermal_field_generator_creation_custom(self):
        """Test creating ThermalFieldGenerator with custom parameters."""
        generator = ThermalFieldGenerator(
            thermal_diffusivity=10.0e-6,
            heat_capacity=600.0,
            density=9000.0,
            ambient_temp=20.0,
            cooling_rate=0.2,
        )

        assert generator.thermal_diffusivity == 10.0e-6
        assert generator.heat_capacity == 600.0
        assert generator.density == 9000.0
        assert generator.ambient_temp == 20.0
        assert generator.cooling_rate == 0.2

    @pytest.mark.unit
    def test_energy_to_temperature(self, generator):
        """Test converting energy density to temperature."""
        energy_density = np.array([1.0e9, 2.0e9, 3.0e9])  # J/m³

        temperature = generator.energy_to_temperature(energy_density, time_step=0.001, voxel_size=0.001)

        assert len(temperature) == len(energy_density)
        assert np.all(temperature > generator.ambient_temp)
        # Higher energy should give higher temperature
        assert temperature[2] > temperature[1] > temperature[0]

    @pytest.mark.unit
    def test_energy_to_temperature_zero_energy(self, generator):
        """Test converting zero energy density."""
        energy_density = np.zeros(5)

        temperature = generator.energy_to_temperature(energy_density)

        # Should be at ambient temperature
        assert np.allclose(temperature, generator.ambient_temp)

    @pytest.mark.unit
    def test_apply_heat_diffusion(self, generator):
        """Test applying heat diffusion."""
        # Create temperature field with hot spot
        temperature_field = np.array([[25.0, 25.0, 25.0], [25.0, 100.0, 25.0], [25.0, 25.0, 25.0]])  # Hot spot

        diffused = generator.apply_heat_diffusion(temperature_field, time_step=0.001, voxel_size=0.001, num_iterations=1)

        assert diffused.shape == temperature_field.shape
        # Heat should spread from hot spot
        assert diffused[1, 1] < temperature_field[1, 1]  # Hot spot cools
        assert diffused[0, 1] > temperature_field[0, 1]  # Neighbors heat up

    @pytest.mark.unit
    def test_apply_heat_diffusion_multiple_iterations(self, generator):
        """Test applying heat diffusion with multiple iterations."""
        temperature_field = np.array([[25.0, 25.0, 25.0], [25.0, 100.0, 25.0], [25.0, 25.0, 25.0]])

        diffused = generator.apply_heat_diffusion(temperature_field, time_step=0.001, voxel_size=0.001, num_iterations=5)

        assert diffused.shape == temperature_field.shape
        # More iterations should spread heat more

    @pytest.mark.unit
    def test_apply_cooling(self, generator):
        """Test applying cooling to temperature field."""
        temperature_field = np.array([100.0, 200.0, 300.0])

        cooled = generator.apply_cooling(temperature_field, time_step=0.1)

        assert len(cooled) == len(temperature_field)
        # All temperatures should decrease toward ambient
        assert np.all(cooled < temperature_field)
        assert np.all(cooled >= generator.ambient_temp)

    @pytest.mark.unit
    def test_apply_cooling_ambient(self, generator):
        """Test cooling when already at ambient temperature."""
        temperature_field = np.array([25.0, 25.0, 25.0])

        cooled = generator.apply_cooling(temperature_field, time_step=0.1)

        # Should remain at ambient
        assert np.allclose(cooled, generator.ambient_temp)

    @pytest.mark.unit
    def test_generate_thermal_field(self, generator):
        """Test generating complete thermal field."""
        energy_density = np.array([[1.0e9, 2.0e9], [3.0e9, 4.0e9]])

        thermal_field = generator.generate_thermal_field(
            energy_density,
            voxel_size=0.001,
            apply_diffusion=True,
            apply_cooling=True,
            time_steps=1,
        )

        assert thermal_field.shape == energy_density.shape
        assert np.all(thermal_field > generator.ambient_temp)

    @pytest.mark.unit
    def test_generate_thermal_field_no_diffusion(self, generator):
        """Test generating thermal field without diffusion."""
        energy_density = np.array([1.0e9, 2.0e9, 3.0e9])

        thermal_field = generator.generate_thermal_field(energy_density, apply_diffusion=False, apply_cooling=False)

        assert len(thermal_field) == len(energy_density)

    @pytest.mark.unit
    def test_generate_thermal_field_multiple_steps(self, generator):
        """Test generating thermal field with multiple time steps."""
        energy_density = np.array([1.0e9, 2.0e9, 3.0e9])

        thermal_field = generator.generate_thermal_field(energy_density, time_steps=5)

        assert len(thermal_field) == len(energy_density)


class TestDensityFieldEstimator:
    """Test suite for DensityFieldEstimator class."""

    @pytest.fixture
    def estimator(self):
        """Create a DensityFieldEstimator instance."""
        return DensityFieldEstimator()

    @pytest.mark.unit
    def test_density_field_estimator_creation_default(self):
        """Test creating DensityFieldEstimator with default parameters."""
        estimator = DensityFieldEstimator()

        assert estimator.base_density == 8000.0
        assert estimator.min_density == 6000.0
        assert estimator.optimal_energy == 50.0

    @pytest.mark.unit
    def test_density_field_estimator_creation_custom(self):
        """Test creating DensityFieldEstimator with custom parameters."""
        estimator = DensityFieldEstimator(base_density=9000.0, min_density=7000.0, optimal_energy=60.0)

        assert estimator.base_density == 9000.0
        assert estimator.min_density == 7000.0
        assert estimator.optimal_energy == 60.0

    @pytest.mark.unit
    def test_estimate_from_energy(self, estimator):
        """Test estimating density from energy density."""
        energy_density = np.array([25.0, 50.0, 75.0, 100.0])  # J/mm³

        density = estimator.estimate_from_energy(energy_density)

        assert len(density) == len(energy_density)
        assert np.all(density >= estimator.min_density)
        assert np.all(density <= estimator.base_density)
        # Optimal energy (50.0) should give higher density
        assert density[1] > density[0]

    @pytest.mark.unit
    def test_estimate_from_energy_low_energy(self, estimator):
        """Test estimating density with low energy."""
        energy_density = np.array([10.0, 20.0, 30.0])

        density = estimator.estimate_from_energy(energy_density)

        # Low energy should give lower density (more porous)
        assert np.all(density < estimator.base_density)
        assert np.all(density >= estimator.min_density)

    @pytest.mark.unit
    def test_estimate_from_energy_high_energy(self, estimator):
        """Test estimating density with high energy (overheating)."""
        energy_density = np.array([100.0, 150.0, 200.0])  # Above optimal

        density = estimator.estimate_from_energy(energy_density)

        # Very high energy may reduce density due to overheating
        assert len(density) == len(energy_density)
        assert np.all(density >= estimator.min_density)

    @pytest.mark.unit
    def test_estimate_from_thermal_history(self, estimator):
        """Test estimating density from thermal history."""
        temperature_field = np.array([500.0, 1000.0, 1500.0, 2000.0])  # °C

        density = estimator.estimate_from_thermal_history(temperature_field, melting_temp=1500.0)

        assert len(density) == len(temperature_field)
        assert np.all(density >= estimator.min_density)
        assert np.all(density <= estimator.base_density)
        # Higher temperatures should give higher density
        assert density[3] > density[2] > density[1] > density[0]

    @pytest.mark.unit
    def test_estimate_from_thermal_history_below_melting(self, estimator):
        """Test estimating density with temperatures below melting."""
        temperature_field = np.array([500.0, 1000.0])  # Below melting

        density = estimator.estimate_from_thermal_history(temperature_field, melting_temp=1500.0)

        assert len(density) == len(temperature_field)
        # Lower temperatures should give lower density
        assert density[0] < density[1]


class TestStressFieldGenerator:
    """Test suite for StressFieldGenerator class."""

    @pytest.fixture
    def generator(self):
        """Create a StressFieldGenerator instance."""
        return StressFieldGenerator()

    @pytest.mark.unit
    def test_stress_field_generator_creation_default(self):
        """Test creating StressFieldGenerator with default parameters."""
        generator = StressFieldGenerator()

        assert generator.thermal_expansion_coeff == 12.0e-6
        assert generator.youngs_modulus == 200.0e9
        assert generator.poissons_ratio == 0.3

    @pytest.mark.unit
    def test_stress_field_generator_creation_custom(self):
        """Test creating StressFieldGenerator with custom parameters."""
        generator = StressFieldGenerator(thermal_expansion_coeff=15.0e-6, youngs_modulus=250.0e9, poissons_ratio=0.25)

        assert generator.thermal_expansion_coeff == 15.0e-6
        assert generator.youngs_modulus == 250.0e9
        assert generator.poissons_ratio == 0.25

    @pytest.mark.unit
    def test_compute_thermal_strain(self, generator):
        """Test computing thermal strain."""
        temperature_field = np.array([25.0, 100.0, 200.0, 300.0])
        reference_temp = 25.0

        strain = generator.compute_thermal_strain(temperature_field, reference_temp)

        assert len(strain) == len(temperature_field)
        # At reference temp, strain should be zero
        assert np.isclose(strain[0], 0.0)
        # Higher temperatures should give higher strain
        assert strain[3] > strain[2] > strain[1] > strain[0]

    @pytest.mark.unit
    def test_compute_thermal_strain_below_reference(self, generator):
        """Test computing thermal strain below reference temperature."""
        temperature_field = np.array([0.0, 10.0, 20.0])
        reference_temp = 25.0

        strain = generator.compute_thermal_strain(temperature_field, reference_temp)

        # Below reference, strain should be negative (contraction)
        assert np.all(strain < 0)

    @pytest.mark.unit
    def test_compute_thermal_stress(self, generator):
        """Test computing thermal stress."""
        temperature_field = np.array([25.0, 100.0, 200.0, 300.0])
        reference_temp = 25.0

        stress = generator.compute_thermal_stress(temperature_field, reference_temp)

        assert len(stress) == len(temperature_field)
        # At reference temp, stress should be zero
        assert np.isclose(stress[0], 0.0)
        # Higher temperatures should give higher stress
        assert stress[3] > stress[2] > stress[1] > stress[0]

    @pytest.mark.unit
    def test_compute_stress_from_gradient(self, generator):
        """Test computing stress from temperature gradients."""
        # Create temperature field with gradient
        temperature_field = np.array([[[25.0, 50.0], [100.0, 150.0]], [[75.0, 100.0], [125.0, 175.0]]])

        stress_components = generator.compute_stress_from_gradient(temperature_field, voxel_size=0.001)

        assert isinstance(stress_components, dict)
        assert "von_mises" in stress_components
        assert "max_principal" in stress_components
        assert "min_principal" in stress_components
        assert stress_components["von_mises"].shape == temperature_field.shape
        assert stress_components["max_principal"].shape == temperature_field.shape
        assert stress_components["min_principal"].shape == temperature_field.shape

    @pytest.mark.unit
    def test_compute_stress_from_gradient_uniform(self, generator):
        """Test computing stress from uniform temperature field."""
        temperature_field = np.ones((3, 3, 3)) * 100.0  # Uniform

        stress_components = generator.compute_stress_from_gradient(temperature_field, voxel_size=0.001)

        # Uniform field should have low stress (no gradients)
        assert "von_mises" in stress_components
        # Von Mises stress should be small for uniform field
        assert np.max(stress_components["von_mises"]) < 1e6  # Small stress

    @pytest.mark.unit
    def test_compute_stress_from_gradient_1d(self, generator):
        """Test computing stress from 1D temperature field."""
        temperature_field = np.array([25.0, 50.0, 100.0, 150.0])

        stress_components = generator.compute_stress_from_gradient(temperature_field, voxel_size=0.001)

        assert "von_mises" in stress_components
        assert stress_components["von_mises"].shape == temperature_field.shape
