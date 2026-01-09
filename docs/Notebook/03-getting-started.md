# Getting Started with AM-QADF Interactive Notebooks

**Version**: 1.0  
**Last Updated**: 2024

## Quick Start Guide

This guide will help you get started with the AM-QADF Interactive Notebooks collection.

## Prerequisites

### Required

- **Python 3.8+**: Check with `python --version`
- **Jupyter Notebook or JupyterLab**: Install with `pip install jupyter` or `pip install jupyterlab`
- **Basic Python knowledge**: Understanding of Python basics

### Recommended

- **MongoDB**: For real data access (optional, notebooks work in demo mode)
- **Virtual Environment**: Isolated Python environment (recommended)

### Optional

- **PyVista**: For advanced 3D visualization
- **scikit-learn**: For ML-based analysis
- **pandas, numpy, matplotlib**: Usually included with Jupyter

## Installation

### Step 1: Clone or Navigate to Repository

```bash
cd AM-QADF/notebooks
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install jupyter ipywidgets matplotlib numpy pandas scipy
```

Optional dependencies:
```bash
pip install pyvista scikit-learn
```

### Step 4: Install AM-QADF Framework (Optional)

If you want to use real framework classes instead of demo mode:

```bash
cd ..
pip install -e .
```

### Step 5: Start Jupyter

```bash
jupyter notebook
# or
jupyter lab
```

## Using the Notebooks

### Step 1: Open a Notebook

1. Navigate to the notebooks directory in Jupyter
2. Click on a notebook (e.g., `00_Introduction_to_AM_QADF.ipynb`)

### Step 2: Run the Setup Cell

1. Find the setup cell (usually the second cell)
2. Click "Run" or press `Shift+Enter`
3. Wait for "✅ Setup complete!" message

### Step 3: Interact with Widgets

1. Scroll to the interactive interface cell
2. Use widgets to:
   - Select modes/methods
   - Adjust parameters
   - Click action buttons
   - View results

### Step 4: Explore and Learn

- Read the introduction and overview
- Try different parameter values
- Observe how results change
- Read the summary at the end

## Notebook Workflow

### Standard Workflow

1. **Read Introduction**: Understand the notebook's purpose
2. **Run Setup**: Execute the setup cell
3. **Explore Interface**: Use widgets to interact
4. **Adjust Parameters**: Try different values
5. **View Results**: Observe visualizations and outputs
6. **Read Summary**: Review key takeaways

### Interactive Workflow

1. **Select Mode**: Choose analysis mode from dropdown
2. **Configure Parameters**: Adjust sliders and checkboxes
3. **Execute Action**: Click action buttons
4. **View Results**: Check center and right panels
5. **Iterate**: Adjust parameters and re-execute

## Demo Mode vs. Real Mode

### Demo Mode (Default)

- Works without framework installation
- Uses synthetic data
- All widgets functional
- Good for learning

### Real Mode

- Requires framework installation
- Uses real MongoDB data
- Full framework capabilities
- Production-ready

### Switching Modes

The notebooks automatically detect framework availability:
- If framework is available → Real mode
- If framework is not available → Demo mode

## Common Tasks

### Running a Complete Notebook

1. Open notebook
2. Run all cells: `Cell → Run All`
3. Interact with widgets
4. Read summary

### Running Specific Cells

1. Select cell
2. Press `Shift+Enter` to run
3. Or click "Run" button

### Resetting a Notebook

1. `Kernel → Restart & Clear Output`
2. Re-run setup cell
3. Continue from there

### Saving Results

- Use export buttons in right panel
- Or copy visualizations manually
- Results are displayed in widgets

## Troubleshooting

### Widgets Not Displaying

**Problem**: Widgets don't appear or are not interactive

**Solutions**:
1. Ensure `ipywidgets` is installed: `pip install ipywidgets`
2. Enable widget extension: `jupyter nbextension enable --py widgetsnbextension`
3. Restart Jupyter
4. Clear browser cache

### Import Errors

**Problem**: Module not found errors

**Solutions**:
1. Install missing packages: `pip install <package-name>`
2. Check Python path
3. Activate virtual environment
4. Restart Jupyter kernel

### Memory Issues

**Problem**: Out of memory errors

**Solutions**:
1. Close other notebooks
2. Restart kernel
3. Use smaller datasets
4. Clear variables: `%reset`

### Slow Performance

**Problem**: Notebooks run slowly

**Solutions**:
1. Reduce data size
2. Use demo mode
3. Close other applications
4. Check system resources

## Best Practices

### For Learning

1. **Start with 00**: Begin with the introduction notebook
2. **Follow Order**: Complete notebooks in numerical order
3. **Take Notes**: Document your learnings
4. **Experiment**: Try different parameter values
5. **Read Summaries**: Review key takeaways

### For Development

1. **Use Real Mode**: Install framework for production use
2. **Customize**: Modify notebooks for your needs
3. **Extend**: Add your own widgets and features
4. **Share**: Contribute improvements

### For Teaching

1. **Progressive Disclosure**: Start with basic notebooks
2. **Interactive Demos**: Use widgets for demonstrations
3. **Assignments**: Use notebooks as assignments
4. **Assessment**: Use summaries for evaluation

## Next Steps

### Recommended Learning Path

1. **Start Here**: [00: Introduction to AM-QADF](04-notebooks/00-introduction.md)
2. **Basics**: Complete notebooks 01-02
3. **Core**: Progress through notebooks 03-05
4. **Advanced**: Continue with notebooks 06-22

### Explore Further

- **[Notebook Structure](02-structure.md)** - Understand organization
- **[Individual Notebooks](04-notebooks/README.md)** - Detailed docs
- **[Widget Documentation](05-widgets/README.md)** - Widget system
- **[Examples](06-examples/README.md)** - Usage examples

## Getting Help

### Documentation

- **[Troubleshooting](07-troubleshooting.md)** - Common issues
- **[Best Practices](08-best-practices.md)** - Best practices
- **[Framework Docs](../AM_QADF/README.md)** - Framework documentation

### Support

- Check error messages in notebook output
- Review troubleshooting guide
- Consult framework documentation
- Check notebook-specific documentation

## Related Documentation

- **[Overview](01-overview.md)** - Notebooks overview
- **[Structure](02-structure.md)** - Notebook organization
- **[Troubleshooting](07-troubleshooting.md)** - Common issues

---

**Last Updated**: 2024

