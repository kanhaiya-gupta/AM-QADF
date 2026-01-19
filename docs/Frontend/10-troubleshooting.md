# Troubleshooting Guide

## Common Issues and Solutions

### Server Issues

#### Server Won't Start

**Problem**: FastAPI server fails to start

**Solutions**:
1. Check if port is already in use:
   ```bash
   # Windows
   netstat -ano | findstr :8000
   
   # Linux/Mac
   lsof -i :8000
   ```

2. Change port in `.env`:
   ```env
   API_PORT=8001
   ```

3. Check Python version:
   ```bash
   python --version  # Should be 3.8+
   ```

4. Verify dependencies:
   ```bash
   pip install -r requirements.txt
   ```

#### Import Errors

**Problem**: `ModuleNotFoundError` or `ImportError`

**Solutions**:
1. Verify Python path includes project root
2. Check if module exists in correct location
3. Reinstall dependencies:
   ```bash
   pip install -r requirements.txt --force-reinstall
   ```

### Database Issues

#### MongoDB Connection Failed

**Problem**: Cannot connect to MongoDB

**Solutions**:
1. Verify MongoDB is running:
   ```bash
   mongosh --eval "db.adminCommand('ping')"
   ```

2. Check connection string in `.env`:
   ```env
   MONGODB_CONNECTION_STRING=mongodb://localhost:27017
   ```

3. Test connection:
   ```python
   from pymongo import MongoClient
   client = MongoClient("mongodb://localhost:27017")
   client.admin.command('ping')
   ```

4. Check network connectivity and firewall settings

#### Database Query Timeout

**Problem**: Queries timeout or take too long

**Solutions**:
1. Add indexes to frequently queried fields
2. Limit result set size with pagination
3. Use projection to return only needed fields
4. Increase timeout in configuration:
   ```env
   QUERY_TIMEOUT_SECONDS=60
   ```

### Template Issues

#### Template Not Found

**Problem**: `TemplateNotFound` error

**Solutions**:
1. Verify template file exists in `client/templates/`
2. Check template path in route handler
3. Verify Jinja2 templates directory configuration

#### Template Rendering Errors

**Problem**: Template syntax errors or rendering fails

**Solutions**:
1. Check Jinja2 syntax in template
2. Verify template variables are passed correctly
3. Check for missing template blocks or includes
4. Enable template auto-reload in development:
   ```python
   templates = Jinja2Templates(
       directory=str(templates_dir),
       auto_reload=True
   )
   ```

### Static File Issues

#### Static Files Not Loading

**Problem**: CSS/JS/images not loading (404 errors)

**Solutions**:
1. Verify static files directory exists: `client/static/`
2. Check StaticFiles mount in `app.py`:
   ```python
   app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
   ```
3. Clear browser cache
4. Check file paths in HTML templates
5. Verify file permissions

#### CSS Not Applying

**Problem**: Styles not appearing

**Solutions**:
1. Check CSS file paths in templates
2. Verify CSS files exist in `client/static/css/`
3. Clear browser cache
4. Check browser console for CSS errors
5. Verify CSS syntax is correct

### API Issues

#### CORS Errors

**Problem**: CORS errors in browser console

**Solutions**:
1. Configure CORS in `app.py`:
   ```python
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["http://localhost:8000"],
       allow_credentials=True,
       allow_methods=["*"],
       allow_headers=["*"],
   )
   ```

2. Check CORS configuration matches your frontend URL

#### Authentication Errors

**Problem**: Authentication fails or tokens invalid

**Solutions**:
1. Verify JWT secret key in `.env`
2. Check token expiration settings
3. Verify token is included in request headers:
   ```javascript
   headers: {
     'Authorization': `Bearer ${token}`
   }
   ```
4. Check token format and validity

#### API Endpoint Not Found

**Problem**: 404 errors for API endpoints

**Solutions**:
1. Verify route is registered in `app.py`
2. Check route path matches request URL
3. Verify router is included with correct prefix
4. Check FastAPI route documentation: `http://localhost:8000/docs`

### Performance Issues

#### Slow Response Times

**Problem**: API responses are slow

**Solutions**:
1. Check database query performance
2. Add indexes to frequently queried fields
3. Implement caching for frequently accessed data
4. Use pagination for large result sets
5. Optimize database queries

#### High Memory Usage

**Problem**: Application uses too much memory

**Solutions**:
1. Limit result set sizes
2. Use pagination
3. Clear caches periodically
4. Optimize data structures
5. Monitor memory usage and identify leaks

### Browser Issues

#### JavaScript Errors

**Problem**: JavaScript errors in browser console

**Solutions**:
1. Check browser console for error messages
2. Verify JavaScript files are loaded
3. Check for syntax errors in JavaScript
4. Verify external dependencies (CDN) are accessible
5. Check browser compatibility

#### Page Not Loading

**Problem**: Page doesn't load or shows blank

**Solutions**:
1. Check browser console for errors
2. Verify server is running
3. Check network tab for failed requests
4. Clear browser cache
5. Try different browser
6. Check server logs for errors

## Debugging Tips

### Enable Debug Mode

Set debug mode in `.env`:
```env
DEBUG=true
LOG_LEVEL=DEBUG
```

### Check Server Logs

Monitor server logs for errors:
```bash
# View logs in real-time
tail -f logs/app.log
```

### Use Browser Developer Tools

- **Console**: Check for JavaScript errors
- **Network**: Monitor API requests and responses
- **Elements**: Inspect HTML structure

### Test API Endpoints

Use FastAPI automatic documentation:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Getting Help

If you're still experiencing issues:

1. Check [Architecture](02-architecture.md) for system design
2. Review [Configuration](08-configuration.md) for settings
3. Check [Performance](09-performance.md) for optimization
4. Review server logs for detailed error messages
5. Check browser console for client-side errors

## Related Documentation

- [Installation](03-installation.md) - Installation guide
- [Configuration](08-configuration.md) - Configuration guide
- [Architecture](02-architecture.md) - System architecture

---

**Next**: [Contributing](11-contributing.md)
