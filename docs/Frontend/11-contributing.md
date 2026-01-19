# Contributing Guide

## Overview

Thank you for your interest in contributing to the AM-QADF Frontend Client! This guide will help you get started.

## Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub
# Clone your fork
git clone https://github.com/your-username/AM-QADF.git
cd AM-QADF
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment

Create `.env` file:
```env
MONGODB_CONNECTION_STRING=mongodb://localhost:27017
API_HOST=0.0.0.0
API_PORT=8000
RELOAD=true
DEBUG=true
```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Changes

- Follow the existing code structure
- Write clear, documented code
- Add comments where necessary
- Follow Python style guidelines (PEP 8)

### 3. Test Your Changes

```bash
# Start the server
python main.py

# Test in browser
# Navigate to http://localhost:8000
```

### 4. Commit Changes

```bash
git add .
git commit -m "Description of your changes"
```

### 5. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub.

## Code Style

### Python Code

- Follow PEP 8 style guide
- Use type hints where appropriate
- Write docstrings for functions and classes
- Keep functions focused and small

### HTML Templates

- Use consistent indentation (2 or 4 spaces)
- Use semantic HTML
- Keep templates simple and readable
- Use template inheritance

### CSS

- Use consistent naming conventions
- Organize CSS by component/module
- Use CSS variables for theming
- Keep styles modular

### JavaScript

- Use modern ES6+ syntax
- Keep functions small and focused
- Add comments for complex logic
- Follow consistent naming conventions

## Project Structure

### Adding a New Module

1. Create module directory in `client/modules/`
2. Add `routes.py` for API routes
3. Create `services/` directory for business logic
4. Add templates in `client/templates/[module]/`
5. Add static files in `client/static/css/modules/[module]/` and `client/static/js/[module]/`
6. Register routes in appropriate layer router

### Adding a New Route

```python
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

router = APIRouter()

@router.get("/", response_class=HTMLResponse)
async def module_page(request: Request):
    return templates.TemplateResponse(
        "module/index.html",
        {"request": request}
    )

@router.get("/api/endpoint")
async def api_endpoint():
    return {"status": "success", "data": {}}
```

## Documentation

### Updating Documentation

- Update relevant documentation files in `docs/Frontend/`
- Keep documentation clear and up-to-date
- Add examples where helpful
- Update diagrams if architecture changes

### Code Comments

- Write clear, concise comments
- Explain "why" not "what"
- Document complex algorithms
- Add docstrings to functions and classes

## Testing

### Manual Testing

- Test all affected functionality
- Test in different browsers
- Test with different screen sizes
- Verify error handling

### Automated Testing

(Add testing framework if available)

## Pull Request Guidelines

### Before Submitting

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] Documentation is updated
- [ ] Code is commented where necessary
- [ ] No console errors or warnings

### Pull Request Description

Include:
- Description of changes
- Why the changes were made
- How to test the changes
- Screenshots (if UI changes)

## Review Process

1. Code will be reviewed by maintainers
2. Feedback will be provided
3. Make requested changes
4. Once approved, changes will be merged

## Questions?

If you have questions:
- Check existing documentation
- Open an issue on GitHub
- Ask in discussions

## Related Documentation

- [Architecture](02-architecture.md) - System architecture
- [Modules](05-modules/README.md) - Module documentation
- [API Reference](06-api-reference/README.md) - API documentation

---

**Thank you for contributing!**
