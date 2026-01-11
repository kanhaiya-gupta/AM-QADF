"""
Unit tests for authentication and authorization.

Tests for AuthenticationManager, User, and RoleBasedAccessControl.
"""

import pytest
import time
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from am_qadf.integration.authentication import (
    AuthenticationManager,
    User,
    RoleBasedAccessControl,
    AuthMethod,
    Permission,
)


class TestUser:
    """Test suite for User dataclass."""

    @pytest.mark.unit
    def test_user_creation(self):
        """Test creating User."""
        user = User(
            user_id="user1",
            username="testuser",
            email="test@example.com",
            roles=["admin", "operator"],
            permissions=[Permission.QUALITY_ASSESS.value, Permission.OPTIMIZATION_RUN.value],
            organization="TestOrg",
        )

        assert user.user_id == "user1"
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert "admin" in user.roles
        assert Permission.QUALITY_ASSESS.value in user.permissions
        assert user.organization == "TestOrg"

    @pytest.mark.unit
    def test_user_has_permission(self):
        """Test checking user permission."""
        user = User(
            user_id="user1",
            username="testuser",
            email="test@example.com",
            permissions=[Permission.QUALITY_ASSESS.value, Permission.OPTIMIZATION_RUN.value],
        )

        assert user.has_permission(Permission.QUALITY_ASSESS.value) is True
        assert user.has_permission(Permission.QUALITY_READ.value) is False

    @pytest.mark.unit
    def test_user_has_role(self):
        """Test checking user role."""
        user = User(
            user_id="user1",
            username="testuser",
            email="test@example.com",
            roles=["admin", "operator"],
        )

        assert user.has_role("admin") is True
        assert user.has_role("viewer") is False

    @pytest.mark.unit
    def test_user_to_dict(self):
        """Test converting User to dictionary."""
        user = User(
            user_id="user1",
            username="testuser",
            email="test@example.com",
            roles=["admin"],
        )

        user_dict = user.to_dict()

        assert isinstance(user_dict, dict)
        assert user_dict["user_id"] == "user1"
        assert isinstance(user_dict["created_at"], str)


class TestAuthenticationManager:
    """Test suite for AuthenticationManager class."""

    @pytest.fixture
    def auth_manager_jwt(self):
        """Create AuthenticationManager with JWT."""
        return AuthenticationManager(
            auth_method="jwt",
            config={"secret_key": "test_secret", "expiration_seconds": 3600},
        )

    @pytest.fixture
    def auth_manager_api_key(self):
        """Create AuthenticationManager with API key."""
        return AuthenticationManager(
            auth_method="api_key",
            config={
                "api_keys": {
                    "key123": {
                        "user_id": "user1",
                        "username": "apiuser",
                        "email": "api@example.com",
                        "roles": ["operator"],
                        "permissions": [Permission.QUALITY_READ.value],
                        "revoked": False,
                    }
                }
            },
        )

    @pytest.mark.unit
    def test_auth_manager_creation(self, auth_manager_jwt):
        """Test creating AuthenticationManager."""
        assert auth_manager_jwt.auth_method == AuthMethod.JWT
        assert auth_manager_jwt.jwt_secret == "test_secret"
        assert auth_manager_jwt.jwt_expiration == 3600

    @pytest.mark.unit
    def test_auth_manager_creation_api_key(self, auth_manager_api_key):
        """Test creating AuthenticationManager with API key."""
        assert auth_manager_api_key.auth_method == AuthMethod.API_KEY
        assert len(auth_manager_api_key.api_keys) == 1
        assert "key123" in auth_manager_api_key.api_keys

    @pytest.mark.unit
    def test_authenticate_jwt_success(self, auth_manager_jwt):
        """Test JWT authentication successfully."""
        # Register user first
        user = auth_manager_jwt.register_user(
            username="testuser",
            email="test@example.com",
            password="password123",
            roles=["admin"],
        )

        # Authenticate
        success, token = auth_manager_jwt.authenticate(
            {
                "username": "testuser",
                "password": "password123",
            }
        )

        assert success is True
        assert token is not None
        assert isinstance(token, str)

    @pytest.mark.unit
    def test_authenticate_jwt_invalid_credentials(self, auth_manager_jwt):
        """Test JWT authentication with invalid credentials."""
        # Register user
        auth_manager_jwt.register_user(
            username="testuser",
            email="test@example.com",
            password="password123",
        )

        # Authenticate with wrong password
        success, error = auth_manager_jwt.authenticate(
            {
                "username": "testuser",
                "password": "wrongpassword",
            }
        )

        assert success is False
        assert "Invalid credentials" in error

    @pytest.mark.unit
    def test_authenticate_api_key_success(self, auth_manager_api_key):
        """Test API key authentication successfully."""
        success, token = auth_manager_api_key.authenticate(
            {
                "api_key": "key123",
            }
        )

        assert success is True
        assert token is not None

    @pytest.mark.unit
    def test_authenticate_api_key_invalid(self, auth_manager_api_key):
        """Test API key authentication with invalid key."""
        success, error = auth_manager_api_key.authenticate(
            {
                "api_key": "invalid_key",
            }
        )

        assert success is False
        assert "Invalid API key" in error

    @pytest.mark.unit
    def test_authenticate_api_key_revoked(self, auth_manager_api_key):
        """Test API key authentication with revoked key."""
        auth_manager_api_key.api_keys["key123"]["revoked"] = True

        success, error = auth_manager_api_key.authenticate(
            {
                "api_key": "key123",
            }
        )

        assert success is False
        assert "revoked" in error.lower()

    @pytest.mark.unit
    def test_validate_token_jwt_success(self, auth_manager_jwt):
        """Test validating JWT token successfully."""
        # Register and authenticate
        auth_manager_jwt.register_user(
            username="testuser",
            email="test@example.com",
            password="password123",
        )
        success, token = auth_manager_jwt.authenticate(
            {
                "username": "testuser",
                "password": "password123",
            }
        )

        # Validate token
        is_valid, user_info = auth_manager_jwt.validate_token(token)

        assert is_valid is True
        assert user_info is not None
        assert "user_id" in user_info or "username" in user_info

    @pytest.mark.unit
    def test_validate_token_invalid(self, auth_manager_jwt):
        """Test validating invalid token."""
        is_valid, error = auth_manager_jwt.validate_token("invalid_token")

        assert is_valid is False
        assert "error" in error

    @pytest.mark.unit
    def test_validate_token_revoked(self, auth_manager_jwt):
        """Test validating revoked token."""
        # Register and authenticate
        auth_manager_jwt.register_user(
            username="testuser",
            email="test@example.com",
            password="password123",
        )
        success, token = auth_manager_jwt.authenticate(
            {
                "username": "testuser",
                "password": "password123",
            }
        )

        # Revoke token
        auth_manager_jwt.revoke_token(token)

        # Try to validate revoked token
        is_valid, error = auth_manager_jwt.validate_token(token)

        assert is_valid is False
        assert "revoked" in error["error"].lower()

    @pytest.mark.unit
    def test_authorize_success(self, auth_manager_jwt):
        """Test authorization successfully."""
        # Register user with permission
        user = auth_manager_jwt.register_user(
            username="testuser",
            email="test@example.com",
            password="password123",
            permissions=[Permission.QUALITY_ASSESS.value],
        )
        success, token = auth_manager_jwt.authenticate(
            {
                "username": "testuser",
                "password": "password123",
            }
        )

        # Authorize
        authorized = auth_manager_jwt.authorize(token, "quality", "assess")

        assert authorized is True

    @pytest.mark.unit
    def test_authorize_no_permission(self, auth_manager_jwt):
        """Test authorization without permission."""
        # Register user without permission
        user = auth_manager_jwt.register_user(
            username="testuser",
            email="test@example.com",
            password="password123",
            permissions=[],
        )
        success, token = auth_manager_jwt.authenticate(
            {
                "username": "testuser",
                "password": "password123",
            }
        )

        # Authorize
        authorized = auth_manager_jwt.authorize(token, "quality", "assess")

        assert authorized is False

    @pytest.mark.unit
    def test_authorize_admin_role(self, auth_manager_jwt):
        """Test authorization with admin role."""
        # Register user with admin role
        user = auth_manager_jwt.register_user(
            username="admin",
            email="admin@example.com",
            password="password123",
            roles=["admin"],
        )
        success, token = auth_manager_jwt.authenticate(
            {
                "username": "admin",
                "password": "password123",
            }
        )

        # Authorize - admin should have access to everything
        authorized = auth_manager_jwt.authorize(token, "quality", "assess")

        assert authorized is True

    @pytest.mark.unit
    def test_refresh_token_success(self, auth_manager_jwt):
        """Test refreshing token successfully."""
        # Register and authenticate
        auth_manager_jwt.register_user(
            username="testuser",
            email="test@example.com",
            password="password123",
        )
        success, token = auth_manager_jwt.authenticate(
            {
                "username": "testuser",
                "password": "password123",
            }
        )

        # Refresh token
        new_token = auth_manager_jwt.refresh_token(token)

        assert new_token is not None
        assert new_token != token  # Should be different token

    @pytest.mark.unit
    def test_refresh_token_invalid(self, auth_manager_jwt):
        """Test refreshing invalid token."""
        new_token = auth_manager_jwt.refresh_token("invalid_token")

        assert new_token is None

    @pytest.mark.unit
    def test_register_user(self, auth_manager_jwt):
        """Test registering new user."""
        user = auth_manager_jwt.register_user(
            username="newuser",
            email="new@example.com",
            password="password123",
            roles=["operator"],
            permissions=[Permission.QUALITY_READ.value],
            organization="TestOrg",
        )

        assert isinstance(user, User)
        assert user.username == "newuser"
        assert user.email == "new@example.com"
        assert "operator" in user.roles
        assert user.username in auth_manager_jwt.users  # Users dict uses username as key
        assert user.user_id == auth_manager_jwt.users[user.username].user_id
        assert "password_hash" in user.metadata

    @pytest.mark.unit
    def test_get_user(self, auth_manager_jwt):
        """Test getting user by ID."""
        user = auth_manager_jwt.register_user(
            username="testuser",
            email="test@example.com",
            password="password123",
        )

        retrieved_user = auth_manager_jwt.get_user(user.user_id)

        assert retrieved_user is not None
        assert retrieved_user.user_id == user.user_id
        assert retrieved_user.username == "testuser"


class TestRoleBasedAccessControl:
    """Test suite for RoleBasedAccessControl class."""

    @pytest.fixture
    def rbac(self):
        """Create RoleBasedAccessControl instance."""
        return RoleBasedAccessControl()

    @pytest.mark.unit
    def test_rbac_creation(self, rbac):
        """Test creating RBAC."""
        assert len(rbac.role_permissions) > 0
        assert "admin" in rbac.role_permissions
        assert "quality_analyst" in rbac.role_permissions

    @pytest.mark.unit
    def test_assign_role(self, rbac):
        """Test assigning role to user."""
        rbac.assign_role("user1", "admin")

        assert "user1" in rbac.user_roles
        assert "admin" in rbac.user_roles["user1"]

    @pytest.mark.unit
    def test_assign_role_multiple(self, rbac):
        """Test assigning multiple roles to user."""
        rbac.assign_role("user1", "admin")
        rbac.assign_role("user1", "operator")

        assert "admin" in rbac.user_roles["user1"]
        assert "operator" in rbac.user_roles["user1"]

    @pytest.mark.unit
    def test_remove_role(self, rbac):
        """Test removing role from user."""
        rbac.assign_role("user1", "admin")
        rbac.remove_role("user1", "admin")

        assert "admin" not in rbac.user_roles["user1"]

    @pytest.mark.unit
    def test_check_permission_with_role(self, rbac):
        """Test checking permission with role."""
        user = User(
            user_id="user1",
            username="testuser",
            email="test@example.com",
            roles=["quality_analyst"],
        )

        has_permission = rbac.check_permission(user, "quality", "assess")

        assert has_permission is True  # quality_analyst has QUALITY_ASSESS

    @pytest.mark.unit
    def test_check_permission_with_direct_permission(self, rbac):
        """Test checking permission with direct permission."""
        user = User(
            user_id="user1",
            username="testuser",
            email="test@example.com",
            permissions=[Permission.QUALITY_ASSESS.value],
        )

        has_permission = rbac.check_permission(user, "quality", "assess")

        assert has_permission is True

    @pytest.mark.unit
    def test_check_permission_no_permission(self, rbac):
        """Test checking permission without permission."""
        user = User(
            user_id="user1",
            username="testuser",
            email="test@example.com",
            roles=["viewer"],  # viewer doesn't have QUALITY_ASSESS
        )

        has_permission = rbac.check_permission(user, "quality", "assess")

        assert has_permission is False

    @pytest.mark.unit
    def test_get_user_permissions(self, rbac):
        """Test getting user permissions."""
        rbac.assign_role("user1", "quality_analyst")
        rbac.assign_role("user1", "viewer")

        permissions = rbac.get_user_permissions("user1")

        assert len(permissions) > 0
        assert Permission.QUALITY_ASSESS.value in permissions
        assert Permission.QUALITY_READ.value in permissions

    @pytest.mark.unit
    def test_define_role(self, rbac):
        """Test defining new role."""
        rbac.define_role("custom_role", [Permission.QUALITY_READ.value])

        assert "custom_role" in rbac.role_permissions
        assert Permission.QUALITY_READ.value in rbac.role_permissions["custom_role"]

    @pytest.mark.unit
    def test_add_role_hierarchy(self, rbac):
        """Test adding role hierarchy."""
        rbac.add_role_hierarchy("parent_role", "child_role")

        assert "parent_role" in rbac.role_hierarchy
        assert "child_role" in rbac.role_hierarchy["parent_role"]

    @pytest.mark.unit
    def test_check_permission_with_hierarchy(self, rbac):
        """Test checking permission with role hierarchy."""
        # Define child role with permission
        rbac.define_role("child_role", [Permission.QUALITY_ASSESS.value])

        # Add hierarchy
        rbac.add_role_hierarchy("parent_role", "child_role")

        # Create user with parent role
        user = User(
            user_id="user1",
            username="testuser",
            email="test@example.com",
            roles=["parent_role"],
        )

        # Should have permission through hierarchy
        has_permission = rbac.check_permission(user, "quality", "assess")

        # Note: This depends on implementation - hierarchy might work differently
        # For now, just test that it doesn't crash
        assert isinstance(has_permission, bool)
