# Notebook Troubleshooting Guide

**Version**: 1.0  
**Last Updated**: 2024

## Common Issues and Solutions

This guide covers common issues encountered when using the AM-QADF Interactive Notebooks and their solutions.

## Widget Issues

### Widgets Not Displaying

**Symptoms**:
- Widgets don't appear in notebook
- Empty spaces where widgets should be
- JavaScript errors in browser console

**Solutions**:
1. **Install ipywidgets**:
   ```bash
   pip install ipywidgets
   ```

2. **Enable widget extension** (Jupyter Notebook):
   ```bash
   jupyter nbextension enable --py widgetsnbextension
   ```

3. **Enable widget extension** (JupyterLab):
   ```bash
   jupyter labextension install @jupyter-widgets/jupyterlab-manager
   ```

4. **Restart Jupyter**:
   - Close Jupyter
   - Restart Jupyter server
   - Clear browser cache

### Widgets Not Interactive

**Symptoms**:
- Widgets appear but don't respond to interaction
- No updates when changing values
- Buttons don't trigger actions

**Solutions**:
1. **Check JavaScript console** for errors
2. **Restart kernel**: `Kernel → Restart`
3. **Re-run setup cell**
4. **Check widget version**: `pip show ipywidgets`

### Widget Layout Issues

**Symptoms**:
- Widgets overlap
- Layout breaks
- Panels not aligned

**Solutions**:
1. **Check layout settings** in code
2. **Adjust panel widths** if needed
3. **Use flex layouts** for responsiveness
4. **Check browser zoom level**

## Import Errors

### Module Not Found

**Symptoms**:
- `ModuleNotFoundError` when running setup
- Import errors for framework modules

**Solutions**:
1. **Install missing packages**:
   ```bash
   pip install <package-name>
   ```

2. **Check Python path**:
   ```python
   import sys
   print(sys.path)
   ```

3. **Activate virtual environment** if using one

4. **Install framework** (for real mode):
   ```bash
   pip install -e .
   ```

### Framework Not Available

**Symptoms**:
- Demo mode message appears
- Framework classes not found

**Solutions**:
1. **This is normal** - notebooks work in demo mode
2. **For real mode**, install framework:
   ```bash
   cd AM-QADF
   pip install -e .
   ```

3. **Check MongoDB connection** if using real data

## Performance Issues

### Slow Execution

**Symptoms**:
- Notebooks run slowly
- Long delays between actions
- System becomes unresponsive

**Solutions**:
1. **Use demo mode** (faster than real mode)
2. **Reduce data size** in parameters
3. **Close other notebooks** and applications
4. **Restart kernel** to free memory
5. **Check system resources** (CPU, memory)

### Memory Errors

**Symptoms**:
- `MemoryError` exceptions
- Kernel dies
- System becomes slow

**Solutions**:
1. **Close other notebooks**
2. **Restart kernel**: `Kernel → Restart`
3. **Use smaller datasets**
4. **Clear variables**: `%reset`
5. **Reduce grid resolution**
6. **Use chunked processing**

## Visualization Issues

### Plots Not Displaying

**Symptoms**:
- Matplotlib plots don't appear
- Empty plot areas
- Plot errors

**Solutions**:
1. **Check matplotlib backend**:
   ```python
   import matplotlib
   matplotlib.use('inline')
   ```

2. **Re-run visualization cell**
3. **Check for errors** in output
4. **Restart kernel** if needed

### 3D Visualization Issues

**Symptoms**:
- PyVista plots don't display
- 3D visualization errors

**Solutions**:
1. **Install PyVista**:
   ```bash
   pip install pyvista
   ```

2. **Use fallback** (notebooks have matplotlib fallback)
3. **Check system graphics** capabilities
4. **Use 2D slices** instead of 3D

## Data Issues

### No Data Available

**Symptoms**:
- "No data" messages
- Empty results
- Query returns nothing

**Solutions**:
1. **Check demo mode** - notebooks generate sample data
2. **For real mode**, verify MongoDB connection
3. **Check data filters** in query widgets
4. **Verify data exists** in database

### Data Validation Errors

**Symptoms**:
- Validation errors
- Data format issues
- Type errors

**Solutions**:
1. **Check data format** matches expected
2. **Use validation widgets** in notebooks
3. **Check data types** and ranges
4. **Review error messages** for details

## Jupyter Issues

### Kernel Dies

**Symptoms**:
- Kernel disconnected
- "Kernel died" message
- Notebook becomes unresponsive

**Solutions**:
1. **Restart kernel**: `Kernel → Restart`
2. **Check memory usage**
3. **Reduce data size**
4. **Close other notebooks**
5. **Restart Jupyter** if persistent

### Notebook Not Saving

**Symptoms**:
- Changes not saved
- Save errors

**Solutions**:
1. **Check file permissions**
2. **Save manually**: `File → Save`
3. **Check disk space**
4. **Restart Jupyter**

## Browser Issues

### Widgets Not Loading

**Symptoms**:
- Widgets don't load in browser
- JavaScript errors

**Solutions**:
1. **Clear browser cache**
2. **Try different browser**
3. **Disable browser extensions**
4. **Check browser console** for errors
5. **Update browser** to latest version

### Display Issues

**Symptoms**:
- Layout broken
- Widgets misaligned
- Display glitches

**Solutions**:
1. **Refresh page**
2. **Clear browser cache**
3. **Check browser zoom** (should be 100%)
4. **Try different browser**

## Getting Help

### Check Documentation

- **[Best Practices](08-best-practices.md)** - Best practices
- **[Getting Started](03-getting-started.md)** - Setup guide
- **[Framework Docs](../AM_QADF/10-troubleshooting.md)** - Framework troubleshooting

### Debug Steps

1. **Check error messages** in notebook output
2. **Review setup cell** output
3. **Check widget status** in browser console
4. **Verify dependencies** are installed
5. **Test in demo mode** first

### Report Issues

If issues persist:
1. Note the error message
2. Check which notebook
3. Document steps to reproduce
4. Check system information
5. Report with details

## Related Documentation

- **[Getting Started](03-getting-started.md)** - Setup guide
- **[Best Practices](08-best-practices.md)** - Best practices
- **[Framework Troubleshooting](../AM_QADF/10-troubleshooting.md)** - Framework issues

---

**Last Updated**: 2024

