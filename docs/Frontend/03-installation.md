# Installation Guide

## Prerequisites

### System Requirements

- **Python**: Version 3.8+ (3.10+ recommended)
- **pip**: Python package manager
- **Modern Browser**: Chrome, Firefox, Safari, or Edge (latest versions)
- **MongoDB**: MongoDB server running and accessible (for AM-QADF backend)

### Browser Support

| Browser | Minimum Version |
|---------|----------------|
| Chrome | 90+ |
| Firefox | 88+ |
| Safari | 14+ |
| Edge | 90+ |

## Installation Methods

### Method 1: Development Setup (Recommended for Developers)

1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository-url>
   cd AM-QADF
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**:
   ```bash
   # Copy environment file if it exists
   cp development.env .env
   # Edit .env with your MongoDB connection and other settings
   ```

4. **Start the FastAPI server**:
   ```bash
   python main.py
   # or
   uvicorn client.app:app --reload --host 0.0.0.0 --port 8000
   ```

5. **Open browser**:
   Navigate to `http://localhost:8000`

### Method 2: Production Deployment

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure production environment**:
   - Set environment variables for production
   - Configure MongoDB connection
   - Set up authentication credentials

3. **Run with production server**:
   ```bash
   uvicorn client.app:app --host 0.0.0.0 --port 8000 --workers 4
   ```

4. **Or use with reverse proxy** (nginx, Apache):
   - Configure reverse proxy to forward requests to FastAPI
   - Serve static files through FastAPI or directly through web server

### Method 3: Docker Deployment

1. **Build Docker image** (if Dockerfile exists):
   ```bash
   docker build -t am-qadf-client .
   ```

2. **Run container**:
   ```bash
   docker run -p 8000:8000 am-qadf-client
   ```

## Environment Configuration

### Environment Variables

Create a `.env` file in the project root directory:

```env
# MongoDB Configuration
MONGODB_CONNECTION_STRING=mongodb://localhost:27017
MONGODB_DATABASE=am_qadf_data
MONGODB_USERNAME=admin
MONGODB_PASSWORD=password

# FastAPI Configuration
API_HOST=0.0.0.0
API_PORT=8000
RELOAD=true  # Set to false in production

# Authentication
AUTH_ENABLED=true
JWT_SECRET_KEY=your-secret-key-here
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24

# Feature Flags
ENABLE_3D_VIZ=true
ENABLE_REALTIME=false
ENABLE_ANALYTICS=true

# Logging
LOG_LEVEL=INFO
DEBUG=false
```

### Configuration Files

- **`.env`**: Environment variables (not committed to git)
- **`development.env`**: Development environment variables (example)
- **`client/app.py`**: FastAPI application configuration
- **`requirements.txt`**: Python dependencies

## Dependencies

### Core Python Dependencies

- **FastAPI**: Web framework for building APIs and serving templates
- **Uvicorn**: ASGI server for running FastAPI
- **Jinja2**: Template engine for server-side rendering
- **Pydantic**: Data validation and settings management
- **Python-dotenv**: Environment variable management

### Frontend Dependencies (CDN)

- **Bootstrap 5.1.3**: CSS framework (loaded via CDN)
- **Font Awesome 6.0**: Icon library (loaded via CDN)
- **Google Fonts**: Typography (Roboto font family)
- **Plotly.js**: 2D charts and plots (loaded via CDN or static)
- **Three.js**: 3D rendering (loaded via CDN or static)

### AM-QADF Framework Dependencies

- All AM-QADF framework modules (installed via `requirements.txt`)
- MongoDB driver (pymongo)
- NumPy, SciPy for data processing
- PyVista for 3D visualization (backend)

## Development Setup

### Recommended IDE Setup

1. **VS Code Extensions**:
   - Python extension
   - Jinja2 extension (for template syntax highlighting)
   - HTML/CSS/JavaScript extensions
   - Python linting (pylint, flake8)

2. **Editor Configuration**:
   - `.editorconfig`: Consistent formatting
   - `.flake8`: Python linting rules
   - `pyproject.toml`: Python project configuration

### Development Scripts

Run the application:

```bash
# Development mode (with auto-reload)
python main.py

# Or directly with uvicorn
uvicorn client.app:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn client.app:app --host 0.0.0.0 --port 8000 --workers 4
```

### Project Structure

```
client/
├── app.py                 # FastAPI application
├── __init__.py
├── static/                # Static files (CSS, JS, images)
│   ├── css/
│   │   ├── base/
│   │   ├── modules/
│   │   └── themes/
│   └── js/
├── templates/             # Jinja2 HTML templates
│   ├── base.html
│   ├── index.html
│   └── [module]/         # Module-specific templates
└── modules/               # Application modules
    ├── core/
    ├── data_layer/
    ├── processing_layer/
    ├── application_layer/
    └── system/
```

## Backend Connection

### Verify Backend Connection

1. **Check FastAPI server is running**:
   ```bash
   curl http://localhost:8000/health
   ```

2. **Check MongoDB connection**:
   - Ensure MongoDB is running
   - Verify connection string in `.env`
   - Test connection from Python:
     ```python
     from pymongo import MongoClient
     client = MongoClient("mongodb://localhost:27017")
     client.admin.command('ping')
     ```

3. **Test frontend**:
   - Open browser to `http://localhost:8000`
   - Check browser developer console for errors
   - Verify API endpoints are accessible
   - Test authentication flow

## Troubleshooting Installation

### Common Issues

1. **Python version mismatch**:
   ```bash
   # Check Python version
   python --version  # Should be 3.8+
   
   # Use virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Dependency installation fails**:
   ```bash
   # Upgrade pip first
   pip install --upgrade pip
   
   # Install dependencies
   pip install -r requirements.txt
   
   # If specific package fails, install individually
   pip install fastapi uvicorn jinja2
   ```

3. **Port already in use**:
   ```bash
   # Change port in environment variable
   export API_PORT=8001  # On Windows: set API_PORT=8001
   
   # Or in .env file
   API_PORT=8001
   ```

4. **MongoDB connection errors**:
   - Ensure MongoDB is running: `mongosh --eval "db.adminCommand('ping')"`
   - Check connection string in `.env`
   - Verify network connectivity
   - Check MongoDB authentication credentials

5. **Template rendering errors**:
   - Verify `client/templates/` directory exists
   - Check Jinja2 syntax in templates
   - Ensure template files are properly formatted

6. **Static files not loading**:
   - Verify `client/static/` directory exists
   - Check FastAPI StaticFiles mount in `app.py`
   - Clear browser cache
   - Check browser console for 404 errors

## Next Steps

After installation:

1. **[Quick Start](04-quick-start.md)** - Get started with the frontend
2. **[Configuration](08-configuration.md)** - Configure the application
3. **[Architecture](02-architecture.md)** - Understand the system design

---

**Next**: [Quick Start](04-quick-start.md)
