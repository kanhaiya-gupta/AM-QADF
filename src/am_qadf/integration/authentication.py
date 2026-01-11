"""
Authentication and authorization for industrial access.

Provides authentication managers, RBAC, and permission management.
"""

import logging
import hashlib
import hmac
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple, Set
from enum import Enum

try:
    import jwt

    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    jwt = None

logger = logging.getLogger(__name__)


class AuthMethod(str, Enum):
    """Authentication methods."""

    JWT = "jwt"
    OAUTH2 = "oauth2"
    API_KEY = "api_key"
    LDAP = "ldap"
    KERBEROS = "kerberos"
    BASIC = "basic"


class Permission(str, Enum):
    """Common permissions."""

    # Quality assessment
    QUALITY_ASSESS = "quality:assess"
    QUALITY_READ = "quality:read"
    QUALITY_WRITE = "quality:write"

    # Process optimization
    OPTIMIZATION_RUN = "optimization:run"
    OPTIMIZATION_READ = "optimization:read"

    # Model management
    MODEL_READ = "model:read"
    MODEL_WRITE = "model:write"
    MODEL_DELETE = "model:delete"

    # Streaming
    STREAMING_START = "streaming:start"
    STREAMING_STOP = "streaming:stop"
    STREAMING_READ = "streaming:read"

    # System administration
    ADMIN_CONFIG = "admin:config"
    ADMIN_MONITOR = "admin:monitor"
    ADMIN_USERS = "admin:users"


@dataclass
class User:
    """User information."""

    user_id: str
    username: str
    email: str
    roles: List[str] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)
    organization: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission."""
        return permission in self.permissions or any(role_perm == permission for role_perm in self.permissions)

    def has_role(self, role: str) -> bool:
        """Check if user has specific role."""
        return role in self.roles

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "roles": self.roles,
            "permissions": self.permissions,
            "organization": self.organization,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class AuthenticationManager:
    """
    Authentication and authorization manager.

    Provides:
    - Multiple authentication methods (JWT, OAuth2, API key, LDAP, Kerberos)
    - Token generation and validation
    - Token refresh
    - User management (basic implementation)
    """

    def __init__(
        self,
        auth_method: str = "jwt",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize authentication manager.

        Args:
            auth_method: Authentication method (AuthMethod enum value or string)
            config: Configuration dictionary
                For JWT: {'secret_key': str, 'algorithm': str, 'expiration_seconds': int}
                For API key: {'api_keys': Dict[str, Dict[str, Any]]}
                For OAuth2: {'client_id': str, 'client_secret': str, 'token_url': str}
                For LDAP: {'server': str, 'port': int, 'base_dn': str}
        """
        self.auth_method = AuthMethod(auth_method) if isinstance(auth_method, str) else auth_method
        self.config = config or {}

        # JWT settings
        self.jwt_secret = self.config.get("secret_key", "default-secret-key-change-in-production")
        self.jwt_algorithm = self.config.get("algorithm", "HS256")
        self.jwt_expiration = self.config.get("expiration_seconds", 3600)  # 1 hour

        # API keys storage (in production, use secure storage)
        self.api_keys: Dict[str, Dict[str, Any]] = self.config.get("api_keys", {})

        # User storage (in production, use database)
        self.users: Dict[str, User] = {}

        # Token blacklist (in production, use Redis or database)
        self.token_blacklist: Set[str] = set()

        logger.info(f"AuthenticationManager initialized with method: {auth_method}")

    def authenticate(self, credentials: Dict[str, str]) -> Tuple[bool, Optional[str]]:
        """
        Authenticate user and return token.

        Args:
            credentials: Credentials dictionary
                For JWT/Basic: {'username': str, 'password': str}
                For API key: {'api_key': str}
                For OAuth2: {'code': str} or {'token': str}

        Returns:
            Tuple of (success, token_or_error_message)
        """
        try:
            if self.auth_method == AuthMethod.JWT or self.auth_method == AuthMethod.BASIC:
                username = credentials.get("username")
                password = credentials.get("password")

                if not username or not password:
                    return False, "Username and password required"

                # Verify credentials (in production, verify against database)
                user = self._verify_user_credentials(username, password)
                if not user:
                    return False, "Invalid credentials"

                # Generate token
                token = self._generate_token(user)
                return True, token

            elif self.auth_method == AuthMethod.API_KEY:
                api_key = credentials.get("api_key")
                if not api_key:
                    return False, "API key required"

                # Verify API key
                if api_key not in self.api_keys:
                    return False, "Invalid API key"

                api_key_info = self.api_keys[api_key]
                if api_key_info.get("revoked", False):
                    return False, "API key revoked"

                # Generate token from API key info
                user = User(
                    user_id=api_key_info.get("user_id", api_key),
                    username=api_key_info.get("username", "api_user"),
                    email=api_key_info.get("email", ""),
                    roles=api_key_info.get("roles", []),
                    permissions=api_key_info.get("permissions", []),
                )
                token = self._generate_token(user)
                return True, token

            elif self.auth_method == AuthMethod.OAUTH2:
                # OAuth2 implementation (simplified)
                code = credentials.get("code")
                token = credentials.get("token")

                if token:
                    # Validate existing OAuth2 token
                    return self.validate_token(token)
                elif code:
                    # Exchange code for token (simplified)
                    return False, "OAuth2 code exchange not implemented"
                else:
                    return False, "OAuth2 code or token required"

            elif self.auth_method == AuthMethod.LDAP:
                # LDAP implementation (simplified)
                return False, "LDAP authentication not implemented"

            elif self.auth_method == AuthMethod.KERBEROS:
                # Kerberos implementation (simplified)
                return False, "Kerberos authentication not implemented"

            else:
                return False, f"Unsupported authentication method: {self.auth_method}"

        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False, f"Authentication error: {str(e)}"

    def _verify_user_credentials(self, username: str, password: str) -> Optional[User]:
        """Verify user credentials (basic implementation)."""
        # In production, verify against database with hashed passwords
        user = self.users.get(username)
        if user:
            # Mock password verification (in production, use proper hashing)
            stored_password = user.metadata.get("password_hash")
            if stored_password and stored_password == self._hash_password(password):
                return user
        return None

    def _hash_password(self, password: str) -> str:
        """Hash password (basic implementation)."""
        # In production, use proper password hashing (bcrypt, argon2, etc.)
        return hashlib.sha256(password.encode()).hexdigest()

    def _generate_token(self, user: User) -> str:
        """Generate authentication token."""
        if self.auth_method == AuthMethod.JWT and JWT_AVAILABLE:
            now = datetime.utcnow()
            payload = {
                "user_id": user.user_id,
                "username": user.username,
                "email": user.email,
                "roles": user.roles,
                "permissions": user.permissions,
                "exp": now + timedelta(seconds=self.jwt_expiration),
                "iat": now,
                "jti": str(uuid.uuid4()),  # JWT ID - ensures uniqueness
            }
            return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
        else:
            # Fallback: simple token (not secure, use JWT in production)
            token_data = f"{user.user_id}:{user.username}:{time.time()}"
            return hashlib.sha256(token_data.encode()).hexdigest()

    def validate_token(self, token: str) -> Tuple[bool, Optional[Dict]]:
        """
        Validate authentication token.

        Args:
            token: Authentication token

        Returns:
            Tuple of (is_valid, user_info_dict_or_error)
        """
        # Check blacklist
        if token in self.token_blacklist:
            return False, {"error": "Token has been revoked"}

        try:
            if self.auth_method == AuthMethod.JWT and JWT_AVAILABLE:
                try:
                    payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
                    return True, payload
                except jwt.ExpiredSignatureError:
                    return False, {"error": "Token has expired"}
                except jwt.InvalidTokenError as e:
                    return False, {"error": f"Invalid token: {str(e)}"}
            else:
                # Fallback: simple token validation (not secure)
                return True, {"user_id": "unknown", "username": "unknown"}

        except Exception as e:
            logger.error(f"Token validation error: {e}")
            return False, {"error": f"Token validation error: {str(e)}"}

    def authorize(self, token: str, resource: str, action: str) -> bool:
        """
        Authorize user action on resource.

        Args:
            token: Authentication token
            resource: Resource identifier (e.g., 'quality', 'optimization')
            action: Action identifier (e.g., 'assess', 'read', 'write')

        Returns:
            True if authorized, False otherwise
        """
        is_valid, user_info = self.validate_token(token)
        if not is_valid:
            return False

        # Check permission
        permission = f"{resource}:{action}"
        user_permissions = user_info.get("permissions", [])
        user_roles = user_info.get("roles", [])

        # Check direct permission
        if permission in user_permissions:
            return True

        # Check role-based permissions (would need RBAC lookup)
        # For now, simple check
        if "admin" in user_roles:
            return True

        return False

    def refresh_token(self, token: str) -> Optional[str]:
        """
        Refresh authentication token.

        Args:
            token: Current authentication token

        Returns:
            New token if refresh successful, None otherwise
        """
        is_valid, user_info = self.validate_token(token)
        if not is_valid:
            return None

        # Create new user object from token info
        user = User(
            user_id=user_info.get("user_id", ""),
            username=user_info.get("username", ""),
            email=user_info.get("email", ""),
            roles=user_info.get("roles", []),
            permissions=user_info.get("permissions", []),
        )

        # Generate new token (jti field ensures uniqueness even if timestamp is same)
        new_token = self._generate_token(user)
        return new_token

    def revoke_token(self, token: str) -> bool:
        """
        Revoke token (add to blacklist).

        Args:
            token: Token to revoke

        Returns:
            True if successful
        """
        self.token_blacklist.add(token)
        logger.info(f"Token revoked")
        return True

    def register_user(
        self,
        username: str,
        email: str,
        password: str,
        roles: Optional[List[str]] = None,
        permissions: Optional[List[str]] = None,
        organization: str = "",
    ) -> User:
        """
        Register new user (basic implementation).

        Args:
            username: Username
            email: Email address
            password: Password (will be hashed)
            roles: User roles
            permissions: User permissions
            organization: Organization name

        Returns:
            Created User object
        """
        user = User(
            user_id=f"user_{len(self.users)}",
            username=username,
            email=email,
            roles=roles or [],
            permissions=permissions or [],
            organization=organization,
            metadata={"password_hash": self._hash_password(password)},
        )
        self.users[username] = user
        logger.info(f"User registered: {username}")
        return user

    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        for user in self.users.values():
            if user.user_id == user_id:
                return user
        return None


class RoleBasedAccessControl:
    """
    Role-based access control (RBAC).

    Provides:
    - Role assignment and management
    - Permission checking
    - Role hierarchy support
    """

    def __init__(self):
        """Initialize RBAC."""
        # Role to permissions mapping
        self.role_permissions: Dict[str, List[str]] = {}

        # User to roles mapping (in production, use database)
        self.user_roles: Dict[str, List[str]] = {}

        # Role hierarchy (parent roles inherit child permissions)
        self.role_hierarchy: Dict[str, List[str]] = {}

        # Initialize default roles
        self._initialize_default_roles()

        logger.info("RoleBasedAccessControl initialized")

    def _initialize_default_roles(self):
        """Initialize default roles and permissions."""
        # Admin role
        self.role_permissions["admin"] = [p.value for p in Permission]

        # Quality analyst role
        self.role_permissions["quality_analyst"] = [
            Permission.QUALITY_ASSESS.value,
            Permission.QUALITY_READ.value,
            Permission.QUALITY_WRITE.value,
            Permission.MODEL_READ.value,
        ]

        # Engineer role
        self.role_permissions["engineer"] = [
            Permission.QUALITY_READ.value,
            Permission.OPTIMIZATION_RUN.value,
            Permission.OPTIMIZATION_READ.value,
            Permission.MODEL_READ.value,
            Permission.STREAMING_READ.value,
        ]

        # Operator role
        self.role_permissions["operator"] = [
            Permission.QUALITY_READ.value,
            Permission.STREAMING_READ.value,
        ]

        # Viewer role
        self.role_permissions["viewer"] = [
            Permission.QUALITY_READ.value,
            Permission.OPTIMIZATION_READ.value,
            Permission.MODEL_READ.value,
        ]

    def assign_role(self, user_id: str, role: str) -> None:
        """
        Assign role to user.

        Args:
            user_id: User identifier
            role: Role name
        """
        if user_id not in self.user_roles:
            self.user_roles[user_id] = []

        if role not in self.user_roles[user_id]:
            self.user_roles[user_id].append(role)
            logger.info(f"Assigned role '{role}' to user {user_id}")

    def remove_role(self, user_id: str, role: str) -> None:
        """Remove role from user."""
        if user_id in self.user_roles and role in self.user_roles[user_id]:
            self.user_roles[user_id].remove(role)
            logger.info(f"Removed role '{role}' from user {user_id}")

    def check_permission(self, user: User, resource: str, action: str) -> bool:
        """
        Check if user has permission for action on resource.

        Args:
            user: User object
            resource: Resource identifier
            action: Action identifier

        Returns:
            True if user has permission
        """
        permission = f"{resource}:{action}"

        # Check direct permissions
        if permission in user.permissions:
            return True

        # Check role-based permissions
        for role in user.roles:
            role_perms = self.role_permissions.get(role, [])
            if permission in role_perms:
                return True

            # Check inherited permissions from role hierarchy
            parent_roles = self.role_hierarchy.get(role, [])
            for parent_role in parent_roles:
                parent_perms = self.role_permissions.get(parent_role, [])
                if permission in parent_perms:
                    return True

        return False

    def get_user_permissions(self, user_id: str) -> List[str]:
        """
        Get all permissions for user.

        Args:
            user_id: User identifier

        Returns:
            List of permission strings
        """
        roles = self.user_roles.get(user_id, [])
        permissions = set()

        for role in roles:
            role_perms = self.role_permissions.get(role, [])
            permissions.update(role_perms)

            # Include inherited permissions
            parent_roles = self.role_hierarchy.get(role, [])
            for parent_role in parent_roles:
                parent_perms = self.role_permissions.get(parent_role, [])
                permissions.update(parent_perms)

        return list(permissions)

    def define_role(self, role: str, permissions: List[str]) -> None:
        """
        Define new role with permissions.

        Args:
            role: Role name
            permissions: List of permission strings
        """
        self.role_permissions[role] = permissions
        logger.info(f"Defined role '{role}' with {len(permissions)} permissions")

    def add_role_hierarchy(self, parent_role: str, child_role: str) -> None:
        """
        Add role hierarchy (parent inherits child permissions).

        Args:
            parent_role: Parent role
            child_role: Child role
        """
        if parent_role not in self.role_hierarchy:
            self.role_hierarchy[parent_role] = []

        if child_role not in self.role_hierarchy[parent_role]:
            self.role_hierarchy[parent_role].append(child_role)
            logger.info(f"Added role hierarchy: {parent_role} -> {child_role}")
