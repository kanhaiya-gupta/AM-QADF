# AM-QADF Interactive Notebooks - Overview

**Version**: 1.0  
**Last Updated**: 2024

## Overview

The AM-QADF Interactive Notebooks collection provides a comprehensive, interactive learning environment for exploring and understanding the AM-QADF (Additive Manufacturing Quality Assessment and Data Fusion) framework. All notebooks feature widget-based interfaces that allow users to explore framework capabilities without writing code.

## Core Principle: Interactive Widget-Based Notebooks

**⚠️ CRITICAL: All notebooks use interactive widget-based interfaces.** This is not optional - it's a core requirement.

### What This Means

- ✅ **ipywidgets** for ALL user interactions (mandatory)
- ✅ **Real-time updates** when parameters change (no re-running cells)
- ✅ **Widget-based UI** with organized panels (top/left/center/right/bottom)
- ✅ **No-code interfaces** - users can explore without writing code
- ✅ **Interactive visualizations** with matplotlib and PyVista
- ✅ **Progressive disclosure** - advanced options hidden by default
- ✅ **Intuitive layout** - clear widget organization
- ✅ **Widget classes** - reusable, modular widget components

### Why Interactive Widgets?

Interactive widgets are essential for:

- **User Experience**: Non-technical users can explore the framework
- **Learning**: Interactive exploration enhances understanding
- **Productivity**: Real-time parameter adjustment speeds up experimentation
- **Demonstration**: Better showcase of framework capabilities
- **Research**: Faster iteration and exploration

## Notebook Collection

### Total Notebooks: 23

The collection is organized into 7 categories:

1. **Introduction and Fundamentals** (3 notebooks)
   - Framework introduction
   - Data querying
   - Voxel grid creation

2. **Core Processing** (3 notebooks)
   - Signal mapping
   - Temporal/spatial alignment
   - Data correction

3. **Data Fusion and Quality** (3 notebooks)
   - Multi-source fusion
   - Quality assessment
   - Quality dashboards

4. **Analytics** (4 notebooks)
   - Statistical analysis
   - Sensitivity analysis
   - Process analysis
   - Virtual experiments

5. **Anomaly Detection** (2 notebooks)
   - Detection methods
   - Detection workflows

6. **Visualization and Workflows** (4 notebooks)
   - 3D visualization
   - Interactive widgets
   - Complete workflows
   - Voxel domain orchestration

7. **Advanced Topics** (4 notebooks)
   - Advanced analytics
   - Performance optimization
   - Custom extensions
   - Troubleshooting

## Key Features

### Interactive Widgets

All notebooks feature:
- **Top Panel**: Mode selectors, action buttons
- **Left Panel**: Configuration and parameters
- **Center Panel**: Visualizations and results
- **Right Panel**: Results, statistics, and export options
- **Bottom Panel**: Status, progress, and logs

### Real-Time Updates

- Visualizations update immediately when parameters change
- No need to re-run cells
- Instant feedback on parameter adjustments

### Progressive Learning

- Organized from basic to advanced concepts
- Each notebook builds on previous knowledge
- Clear learning paths for different skill levels

### Comprehensive Coverage

- All framework modules are covered
- Multiple examples per topic
- Real-world use cases

## Widget System

### Standard Widget Layout

```
┌─────────────────────────────────────────────────────────┐
│  Top Panel: Mode Selectors, Action Buttons              │
├──────────┬──────────────────────────────┬──────────────┤
│          │                              │              │
│  Left    │                              │  Right      │
│  Panel:  │    Center Panel:            │  Panel:     │
│  Config  │    Visualizations            │  Results    │
│          │                              │              │
├──────────┴──────────────────────────────┴──────────────┤
│  Bottom Panel: Status, Progress, Logs                   │
└─────────────────────────────────────────────────────────┘
```

### Widget Types

- **Dropdowns**: Mode selection, method selection
- **Sliders**: Parameter adjustment (int/float)
- **Checkboxes**: Option toggles
- **Buttons**: Action triggers
- **Text Areas**: Code display, logs
- **Output Widgets**: Visualizations, results
- **Accordions**: Collapsible configuration sections
- **Tabs**: Multiple views

## Learning Objectives

By completing the notebooks, users will:

1. ✅ Understand AM-QADF framework architecture
2. ✅ Learn to query and access multi-source data
3. ✅ Create and manipulate voxel grids
4. ✅ Map signals to voxel grids using various methods
5. ✅ Align temporal and spatial data
6. ✅ Correct geometric distortions and process signals
7. ✅ Fuse multi-source data
8. ✅ Assess data quality
9. ✅ Perform statistical and sensitivity analysis
10. ✅ Detect anomalies in manufacturing data
11. ✅ Visualize 3D voxel domain data
12. ✅ Build complete workflows
13. ✅ Optimize performance
14. ✅ Create custom extensions
15. ✅ Troubleshoot common issues

## Target Audience

### Primary Users

- **Researchers**: Explore framework capabilities for research
- **Engineers**: Learn framework usage for production
- **Students**: Learn additive manufacturing data analysis
- **Developers**: Understand framework architecture

### Skill Levels

- **Beginner**: Start with notebooks 00-02, 16
- **Intermediate**: Progress through notebooks 03-08
- **Advanced**: Complete notebooks 09-14, 19
- **Expert**: Master notebooks 15, 17-18, 20-22

## Prerequisites

### Required

- Python 3.8+
- Jupyter Notebook or JupyterLab
- Basic understanding of Python

### Recommended

- Familiarity with additive manufacturing
- Understanding of data analysis concepts
- Knowledge of numpy, pandas, matplotlib

### Optional

- MongoDB (for real data access)
- PyVista (for advanced 3D visualization)
- scikit-learn (for ML-based analysis)

## Installation

See [Getting Started](03-getting-started.md) for detailed installation instructions.

## Usage

1. **Start Jupyter**: Launch Jupyter Notebook or JupyterLab
2. **Open Notebook**: Navigate to the notebooks directory
3. **Run Setup Cell**: Execute the setup cell to import dependencies
4. **Interact**: Use widgets to explore framework capabilities
5. **Learn**: Follow the notebook content and examples

## Related Documentation

- **[Framework Documentation](../AM_QADF/README.md)** - AM-QADF framework docs
- **[Notebook Plan](../../notebooks/NOTEBOOK_PLAN.md)** - Original plan
- **[Widget Specifications](../../notebooks/Widget_Specifications.md)** - Widget specs

---

**Last Updated**: 2024

