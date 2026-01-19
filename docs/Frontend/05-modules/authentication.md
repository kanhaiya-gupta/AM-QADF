# Authentication Module

## Overview

The Authentication module handles user authentication, authorization, and session management.

## Features

- **User Login**: Secure login with credentials
- **User Registration**: New user registration
- **JWT Tokens**: Token-based authentication
- **Session Management**: User session handling
- **Password Reset**: Password reset functionality
- **Profile Management**: User profile management
- **Role-Based Access Control**: RBAC support

## Components

### Routes (`routes.py`)

- `GET /auth/login` - Login page
- `POST /api/auth/login` - Authenticate user
- `GET /auth/register` - Registration page
- `POST /api/auth/register` - Register new user
- `GET /auth/profile` - User profile page
- `POST /api/auth/logout` - Logout user
- `POST /api/auth/password-reset` - Request password reset

### Services

- **AuthService**: Authentication operations
- **UserService**: User management

### Templates

- `templates/auth/login.html` - Login page
- `templates/auth/register.html` - Registration page
- `templates/auth/profile.html` - Profile page
- `templates/auth/password-reset.html` - Password reset page

## Usage

### Login

```javascript
const credentials = {
  username: "user@example.com",
  password: "password123"
};

const response = await API.post('/auth/login', credentials);
localStorage.setItem('auth_token', response.data.token);
```

### Authenticated Request

```javascript
// Token is automatically included via API client
const data = await API.get('/api/data-query/models');
```

## Related Documentation

- [Authentication API Reference](../06-api-reference/authentication-api.md)
- [Architecture](../02-architecture.md)

---

**Parent**: [Modules](README.md)
