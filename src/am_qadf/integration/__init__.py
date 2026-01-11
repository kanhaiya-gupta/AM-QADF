"""
Integration module for industrial system integration.

Provides:
- MPM (Manufacturing Process Management) system integration
- Manufacturing equipment integration
- REST API gateway
- Authentication and authorization
"""

from am_qadf.integration.mpm_integration import (
    MPMClient,
    MPMProcessData,
    MPMStatus,
)

from am_qadf.integration.manufacturing_systems import (
    EquipmentClient,
    EquipmentStatus,
    EquipmentType,
    EquipmentStatusValue,
)

from am_qadf.integration.api_gateway import (
    APIGateway,
    APIEndpoint,
    APIMiddleware,
    APIRequest,
    APIResponse,
    HTTPMethod,
)

from am_qadf.integration.authentication import (
    AuthenticationManager,
    User,
    RoleBasedAccessControl,
    AuthMethod,
    Permission,
)

__all__ = [
    # MPM Integration
    "MPMClient",
    "MPMProcessData",
    "MPMStatus",
    # Manufacturing Systems
    "EquipmentClient",
    "EquipmentStatus",
    "EquipmentType",
    "EquipmentStatusValue",
    # API Gateway
    "APIGateway",
    "APIEndpoint",
    "APIMiddleware",
    "APIRequest",
    "APIResponse",
    "HTTPMethod",
    # Authentication
    "AuthenticationManager",
    "User",
    "RoleBasedAccessControl",
    "AuthMethod",
    "Permission",
]
