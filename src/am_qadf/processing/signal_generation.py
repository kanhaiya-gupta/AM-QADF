"""
Signal Generation

Generate derived signals from base laser parameters:
- Thermal fields from energy density
- Density field estimation
- Stress fields from thermal gradients
"""

from typing import Optional, Tuple, Dict, Any
import numpy as np
from scipy import ndimage
from scipy.spatial.distance import cdist


class ThermalFieldGenerator:
    """
    Generate thermal fields from energy density using heat diffusion models.

    Converts laser energy input into temperature distributions,
    accounting for heat conduction and cooling.
    """

    def __init__(
        self,
        thermal_diffusivity: float = 8.0e-6,  # m²/s (typical for metals)
        heat_capacity: float = 500.0,  # J/(kg·K)
        density: float = 8000.0,  # kg/m³
        ambient_temp: float = 25.0,  # °C
        cooling_rate: float = 0.1,  # 1/s
    ):
        """
        Initialize thermal field generator.

        Args:
            thermal_diffusivity: Thermal diffusivity of material (m²/s)
            heat_capacity: Specific heat capacity (J/(kg·K))
            density: Material density (kg/m³)
            ambient_temp: Ambient temperature (°C)
            cooling_rate: Cooling rate constant (1/s)
        """
        self.thermal_diffusivity = thermal_diffusivity
        self.heat_capacity = heat_capacity
        self.density = density
        self.ambient_temp = ambient_temp
        self.cooling_rate = cooling_rate

    def energy_to_temperature(
        self,
        energy_density: np.ndarray,
        time_step: float = 0.001,  # seconds
        voxel_size: float = 0.001,  # meters
    ) -> np.ndarray:
        """
        Convert energy density to temperature using simple heat model.

        Args:
            energy_density: Energy density array (J/m³)
            time_step: Time step for heat diffusion (seconds)
            voxel_size: Size of each voxel (meters)

        Returns:
            Temperature array (°C)
        """
        # Convert energy density to temperature rise
        # Q = m * c * ΔT, so ΔT = Q / (m * c)
        # For a voxel: m = ρ * V, where V = voxel_size³
        voxel_volume = voxel_size**3
        voxel_mass = self.density * voxel_volume

        # Temperature rise from energy input
        delta_temp = energy_density * voxel_volume / (voxel_mass * self.heat_capacity)

        # Add ambient temperature
        temperature = self.ambient_temp + delta_temp

        return temperature

    def apply_heat_diffusion(
        self,
        temperature_field: np.ndarray,
        time_step: float = 0.001,
        voxel_size: float = 0.001,
        num_iterations: int = 1,
    ) -> np.ndarray:
        """
        Apply heat diffusion to temperature field.

        Uses finite difference method to simulate heat conduction.

        Args:
            temperature_field: Initial temperature field (°C)
            time_step: Time step for diffusion (seconds)
            voxel_size: Size of each voxel (meters)
            num_iterations: Number of diffusion iterations

        Returns:
            Diffused temperature field (°C)
        """
        dt = time_step
        dx = voxel_size
        alpha = self.thermal_diffusivity

        # Diffusion coefficient: D = alpha * dt / dx²
        diffusion_coeff = alpha * dt / (dx**2)

        # Apply diffusion using Gaussian filter (approximation)
        # For stability, diffusion_coeff should be < 0.5
        if diffusion_coeff > 0.5:
            # Reduce time step or increase iterations
            num_iterations = int(np.ceil(diffusion_coeff * 2))
            diffusion_coeff = 0.5

        result = temperature_field.copy()

        for _ in range(num_iterations):
            # Apply Gaussian smoothing as approximation to heat diffusion
            sigma = np.sqrt(2 * diffusion_coeff)
            result = ndimage.gaussian_filter(result, sigma=sigma, mode="nearest")

        return result

    def apply_cooling(self, temperature_field: np.ndarray, time_step: float = 0.001) -> np.ndarray:
        """
        Apply cooling to temperature field (exponential decay to ambient).

        Args:
            temperature_field: Temperature field (°C)
            time_step: Time step (seconds)

        Returns:
            Cooled temperature field (°C)
        """
        # Exponential cooling: T(t) = T_amb + (T_0 - T_amb) * exp(-k*t)
        temp_above_ambient = temperature_field - self.ambient_temp
        cooled_temp = self.ambient_temp + temp_above_ambient * np.exp(-self.cooling_rate * time_step)

        return cooled_temp

    def generate_thermal_field(
        self,
        energy_density: np.ndarray,
        voxel_size: float = 0.001,
        apply_diffusion: bool = True,
        apply_cooling: bool = True,
        time_steps: int = 1,
    ) -> np.ndarray:
        """
        Generate complete thermal field from energy density.

        Args:
            energy_density: Energy density array (J/m³)
            voxel_size: Size of each voxel (meters)
            apply_diffusion: Whether to apply heat diffusion
            apply_cooling: Whether to apply cooling
            time_steps: Number of time steps to simulate

        Returns:
            Thermal field (temperature in °C)
        """
        # Convert energy to initial temperature
        time_step = 0.001  # 1 ms
        temperature = self.energy_to_temperature(energy_density, time_step, voxel_size)

        # Apply diffusion and cooling over time steps
        for _ in range(time_steps):
            if apply_diffusion:
                temperature = self.apply_heat_diffusion(temperature, time_step=time_step, voxel_size=voxel_size)

            if apply_cooling:
                temperature = self.apply_cooling(temperature, time_step=time_step)

        return temperature


class DensityFieldEstimator:
    """
    Estimate density field from process parameters.

    Uses energy density and thermal history to estimate
    material density (porosity, consolidation).
    """

    def __init__(
        self,
        base_density: float = 8000.0,  # kg/m³ (fully dense)
        min_density: float = 6000.0,  # kg/m³ (porous)
        optimal_energy: float = 50.0,  # J/mm³ (optimal energy density)
    ):
        """
        Initialize density field estimator.

        Args:
            base_density: Base material density when fully dense (kg/m³)
            min_density: Minimum density (porous material) (kg/m³)
            optimal_energy: Optimal energy density for full consolidation (J/mm³)
        """
        self.base_density = base_density
        self.min_density = min_density
        self.optimal_energy = optimal_energy

    def estimate_from_energy(self, energy_density: np.ndarray) -> np.ndarray:
        """
        Estimate density from energy density.

        Uses a sigmoid function: density increases with energy up to optimal,
        then may decrease slightly (overheating).

        Args:
            energy_density: Energy density array (J/mm³ or J/m³)

        Returns:
            Estimated density array (kg/m³)
        """
        # Normalize energy density
        normalized_energy = energy_density / self.optimal_energy

        # Sigmoid function: density increases with energy
        # At low energy: density ≈ min_density
        # At optimal energy: density ≈ base_density
        # At very high energy: density may decrease slightly (overheating)

        # Simple model: density = min + (base - min) * sigmoid(energy/optimal)
        sigmoid = 1.0 / (1.0 + np.exp(-2.0 * (normalized_energy - 1.0)))

        density = self.min_density + (self.base_density - self.min_density) * sigmoid

        # Overheating effect: very high energy reduces density
        overheating_factor = np.where(
            normalized_energy > 2.0,
            1.0 - 0.1 * (normalized_energy - 2.0),  # 10% reduction per unit above 2x optimal
            1.0,
        )
        overheating_factor = np.clip(overheating_factor, 0.5, 1.0)

        density = density * overheating_factor

        return density

    def estimate_from_thermal_history(self, temperature_field: np.ndarray, melting_temp: float = 1500.0) -> np.ndarray:  # °C
        """
        Estimate density from thermal history (temperature field).

        Higher temperatures (near melting) indicate better consolidation.

        Args:
            temperature_field: Temperature field (°C)
            melting_temp: Melting temperature (°C)

        Returns:
            Estimated density array (kg/m³)
        """
        # Normalize temperature relative to melting point
        # Use a sigmoid-like function to ensure strictly increasing density
        # Don't clip to allow values above melting_temp to continue increasing
        normalized_temp = temperature_field / melting_temp

        # Use a function that ensures strictly increasing density with temperature
        # For values <= 1.0: use power law, for values > 1.0: continue increasing
        # density = min + (base - min) * sigmoid(normalized_temp)
        # Using tanh to ensure smooth transition and strictly increasing
        sigmoid_factor = np.tanh(normalized_temp)
        # Scale to ensure we reach base_density at melting_temp
        # At normalized_temp=1.0, tanh(1.0) ≈ 0.76, so we need to scale
        # We want sigmoid_factor=1.0 when normalized_temp=1.0
        # Use: sigmoid_factor = tanh(normalized_temp * 1.5) to get closer to 1.0 at temp=1.0
        sigmoid_factor = np.tanh(normalized_temp * 1.5)

        # Density increases with temperature (better consolidation near melting)
        density = self.min_density + (self.base_density - self.min_density) * sigmoid_factor

        # Ensure density is within bounds
        density = np.clip(density, self.min_density, self.base_density)

        return density


class StressFieldGenerator:
    """
    Generate stress fields from thermal gradients.

    Thermal expansion and contraction create residual stresses.
    """

    def __init__(
        self,
        thermal_expansion_coeff: float = 12.0e-6,  # 1/K
        youngs_modulus: float = 200.0e9,  # Pa
        poissons_ratio: float = 0.3,
    ):
        """
        Initialize stress field generator.

        Args:
            thermal_expansion_coeff: Coefficient of thermal expansion (1/K)
            youngs_modulus: Young's modulus (Pa)
            poissons_ratio: Poisson's ratio
        """
        self.thermal_expansion_coeff = thermal_expansion_coeff
        self.youngs_modulus = youngs_modulus
        self.poissons_ratio = poissons_ratio

    def compute_thermal_strain(self, temperature_field: np.ndarray, reference_temp: float = 25.0) -> np.ndarray:  # °C
        """
        Compute thermal strain from temperature field.

        Args:
            temperature_field: Temperature field (°C)
            reference_temp: Reference temperature (°C)

        Returns:
            Thermal strain (dimensionless)
        """
        delta_temp = temperature_field - reference_temp
        thermal_strain = self.thermal_expansion_coeff * delta_temp

        return thermal_strain

    def compute_thermal_stress(self, temperature_field: np.ndarray, reference_temp: float = 25.0) -> np.ndarray:
        """
        Compute thermal stress from temperature field.

        Uses: σ = E * α * ΔT (for constrained thermal expansion)

        Args:
            temperature_field: Temperature field (°C)
            reference_temp: Reference temperature (°C)

        Returns:
            Thermal stress (Pa)
        """
        thermal_strain = self.compute_thermal_strain(temperature_field, reference_temp)
        thermal_stress = self.youngs_modulus * thermal_strain

        return thermal_stress

    def compute_stress_from_gradient(
        self, temperature_field: np.ndarray, voxel_size: float = 0.001  # meters
    ) -> Dict[str, np.ndarray]:
        """
        Compute stress components from temperature gradients.

        Args:
            temperature_field: Temperature field (°C)
            voxel_size: Size of each voxel (meters)

        Returns:
            Dictionary with stress components:
            - 'von_mises': Von Mises equivalent stress (Pa)
            - 'max_principal': Maximum principal stress (Pa)
            - 'min_principal': Minimum principal stress (Pa)
        """
        ndim = temperature_field.ndim

        # Compute temperature gradients based on dimensionality
        if ndim == 1:
            # 1D: only x-gradient
            grad_x = np.gradient(temperature_field) / voxel_size
            grad_y = np.zeros_like(grad_x)
            grad_z = np.zeros_like(grad_x)
        elif ndim == 2:
            # 2D: x and y gradients
            grad_x = np.gradient(temperature_field, axis=0) / voxel_size
            grad_y = np.gradient(temperature_field, axis=1) / voxel_size
            grad_z = np.zeros_like(grad_x)
        else:
            # 3D: x, y, and z gradients
            grad_x = np.gradient(temperature_field, axis=0) / voxel_size
            grad_y = np.gradient(temperature_field, axis=1) / voxel_size
            grad_z = np.gradient(temperature_field, axis=2) / voxel_size

        # Convert gradients to thermal strains
        # Strain gradient ≈ thermal expansion * temperature gradient
        strain_grad_x = self.thermal_expansion_coeff * grad_x
        strain_grad_y = self.thermal_expansion_coeff * grad_y
        strain_grad_z = self.thermal_expansion_coeff * grad_z

        # Convert to stress (simplified: σ = E * ε)
        stress_xx = self.youngs_modulus * strain_grad_x
        stress_yy = self.youngs_modulus * strain_grad_y
        stress_zz = self.youngs_modulus * strain_grad_z

        # Von Mises equivalent stress
        # For 1D/2D, use appropriate formula
        if ndim == 1:
            # For 1D, von Mises is just the absolute stress
            von_mises = np.abs(stress_xx)
        else:
            # For 2D/3D, use full von Mises formula
            von_mises = np.sqrt(
                0.5 * ((stress_xx - stress_yy) ** 2 + (stress_yy - stress_zz) ** 2 + (stress_zz - stress_xx) ** 2)
            )

        # Principal stresses (simplified: use max/min of diagonal components)
        max_principal = np.maximum.reduce([stress_xx, stress_yy, stress_zz])
        min_principal = np.minimum.reduce([stress_xx, stress_yy, stress_zz])

        return {
            "von_mises": von_mises,
            "max_principal": max_principal,
            "min_principal": min_principal,
        }
