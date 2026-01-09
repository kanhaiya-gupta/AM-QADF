# Troubleshooting Guide

## Common Issues

### MongoDB Connection Errors

**Problem**: Cannot connect to MongoDB

**Solution**:
```python
# Check connection string
mongodb_client = MongoDBClient("mongodb://localhost:27017")

# Verify MongoDB is running
# docker ps | grep mongo
```

### Memory Issues

**Problem**: Out of memory with large voxel grids

**Solution**:
- Use adaptive resolution grids
- Process data in chunks
- Use Spark for distributed processing

### Import Errors

**Problem**: Module not found

**Solution**:
```bash
# Reinstall package
pip install -e .

# Check Python path
python -c "import sys; print(sys.path)"
```

## Getting Help

- Check [Examples](07-examples/README.md) for usage patterns
- Review [API Reference](06-api-reference/README.md) for correct usage
- See [Modules](05-modules/README.md) for module-specific issues

## Related

- [Installation](03-installation.md) - Installation troubleshooting
- [Configuration](08-configuration.md) - Configuration issues

---

**Parent**: [Framework Documentation](README.md)

