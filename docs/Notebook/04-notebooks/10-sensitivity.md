# Notebook 10: Sensitivity Analysis

**File**: `10_Sensitivity_Analysis.ipynb`  
**Category**: Analytics  
**Duration**: 90-120 minutes

## Purpose

This notebook teaches you how to perform sensitivity analysis to understand parameter influence on build quality. You'll learn global methods (Sobol, Morris), local methods, design of experiments, and uncertainty quantification.

## Learning Objectives

By the end of this notebook, you will:

- ✅ Perform global sensitivity analysis (Sobol, Morris, FAST)
- ✅ Conduct local sensitivity analysis
- ✅ Design experiments (Factorial, LHS, Random, Grid)
- ✅ Quantify uncertainty (Monte Carlo, Bayesian)
- ✅ Interpret sensitivity indices and rankings

## Topics Covered

### Global Sensitivity Methods

- **Sobol Indices**: Variance-based sensitivity
- **Morris Screening**: Elementary effects method
- **FAST**: Fourier Amplitude Sensitivity Test
- **RBD**: Random Balance Design

### Local Sensitivity Methods

- **Derivatives**: Partial derivatives
- **Perturbation**: Parameter perturbation
- **Central Differences**: Numerical differentiation

### Design of Experiments

- **Factorial Design**: Full factorial experiments
- **LHS**: Latin Hypercube Sampling
- **Random Sampling**: Random parameter sampling
- **Grid Sampling**: Grid-based sampling

### Uncertainty Quantification

- **Monte Carlo**: Monte Carlo simulation
- **Bayesian**: Bayesian uncertainty quantification
- **Taylor Expansion**: Taylor series approximation

## Interactive Widgets

### Top Panel

- **Sensitivity Method**: Dropdown (Sobol/Morris/Local/DoE/Uncertainty)
- **Model Selector**: Dropdown to select model
- **Execute Analysis**: Button to execute analysis
- **Compare Methods**: Button to compare methods

### Left Panel

- **Problem Definition**: Parameter ranges, model selection
- **Dynamic Configuration**: Accordion sections
  - **Sobol**: Sample size, confidence level
  - **Morris**: Trajectories, levels
  - **Local**: Perturbation size, method
  - **DoE**: Design type, sample size
  - **Uncertainty**: Quantification method, samples

### Center Panel

- **Visualization Modes**: Radio buttons
  - **Indices**: Sensitivity indices visualization
  - **Rankings**: Parameter rankings
  - **Comparison**: Method comparison
  - **Uncertainty**: Uncertainty visualization

### Right Panel

- **Sensitivity Indices**: Indices display
- **Parameter Rankings**: Ranked parameter list
- **Method Performance**: Performance metrics
- **Uncertainty Results**: Uncertainty quantification results
- **Export Options**: Export analysis results

### Bottom Panel

- **Status Display**: Current operation status
- **Progress Bar**: Analysis progress
- **Info Display**: Additional information

## Usage

### Step 1: Define Problem

1. Select model for analysis
2. Define parameter ranges
3. Set output variables

### Step 2: Select Method

1. Choose sensitivity method
2. Configure method parameters
3. Set sample sizes

### Step 3: Execute Analysis

1. Click "Execute Analysis" button
2. Wait for analysis to complete
3. Review sensitivity indices

### Step 4: Interpret Results

1. View sensitivity indices
2. Check parameter rankings
3. Compare methods
4. Analyze uncertainty
5. Export results

## Example Workflow

1. **Select Method**: Choose "Sobol"
2. **Define Parameters**: Set laser power and scan speed ranges
3. **Configure**: Set sample size to 1000
4. **Execute**: Run Sobol analysis
5. **Review**: Check sensitivity indices
6. **Compare**: Compare with Morris method

## Key Takeaways

1. **Method Selection**: Choose appropriate method
2. **Parameter Ranges**: Define realistic parameter ranges
3. **Sample Size**: Use adequate sample sizes
4. **Interpretation**: Understand sensitivity indices
5. **Comparison**: Compare multiple methods

## Related Notebooks

- **Previous**: [09: Statistical Analysis](09-statistical.md)
- **Next**: [11: Process Analysis and Optimization](11-process-analysis.md)
- **Related**: [19: Advanced Analytics Workflow](19-advanced-analytics.md)

## Related Documentation

- **[Analytics Module](../../AM_QADF/05-modules/analytics.md)** - Analytics details
- **[Sensitivity Analysis](../../AM_QADF/05-modules/analytics.md#sensitivity-analysis)** - Sensitivity analysis

---

**Last Updated**: 2024

