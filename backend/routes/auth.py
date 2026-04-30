"""
Authentication and authorization helpers.
"""
from datetime import datetime, timedelta
from typing import Callable
import os
import secrets
import string
import sys

import bcrypt
from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings
from database import get_database
from models.schemas import (
    PasswordChangeRequest,
    TokenResponse,
    UserLogin,
    UserRegister,
    UserResponse,
    UserRole,
)

router = APIRouter(prefix="/api/auth", tags=["Authentication"])
security = HTTPBearer()


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    pwd_bytes = password.encode("utf-8")[:72]
    return bcrypt.hashpw(pwd_bytes, bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against a bcrypt hash."""
    pwd_bytes = password.encode("utf-8")[:72]
    return bcrypt.checkpw(pwd_bytes, hashed.encode("utf-8"))


def generate_temporary_password(length: int = 12) -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def _normalize_role(role: str | None) -> str:
    return "nurse" if role == "caregiver" else (role or "")


def create_token(user_id: str, role: str, org_id: str | None = None) -> str:
    """Create JWT token."""
    payload = {
        "sub": user_id,
        "role": _normalize_role(role),
        "org_id": org_id,
        "exp": datetime.utcnow() + timedelta(hours=settings.JWT_EXPIRATION_HOURS),
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def _org_name(db, org_id: str | None) -> str | None:
    if not org_id:
        return None
    try:
        try:
            org = db.organizations.find_one({"_id": ObjectId(org_id)})
        except Exception:
            org = None
    except Exception:
        return None
    return org.get("name") if org else None


def user_response(user: dict, db=None) -> UserResponse:
    if db is None:
        db = get_database()
    role = _normalize_role(user.get("role"))
    return UserResponse(
        id=str(user["_id"]),
        name=user.get("name", ""),
        email=user.get("email", ""),
        role=role,
        org_id=user.get("org_id"),
        org_name=_org_name(db, user.get("org_id")),
        status=user.get("status", "active"),
        must_change_password=bool(user.get("must_change_password", False)),
        created_at=user.get("created_at"),
    )


def _load_current_user_from_id(user_id: str) -> dict:
    db = get_database()
    try:
        user = db.users.find_one({"_id": ObjectId(user_id)})
    except Exception:
        user = None
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    role = _normalize_role(user.get("role"))
    if user.get("role") == "caregiver":
        db.users.update_one({"_id": user["_id"]}, {"$set": {"role": role}})
        user["role"] = role

    if user.get("status", "active") == "disabled":
        raise HTTPException(status_code=403, detail="Account disabled")

    org_id = user.get("org_id")
    org_name = None
    if role != UserRole.SUPER_ADMIN.value:
        if not org_id:
            raise HTTPException(status_code=403, detail="User is not assigned to an organization")
        try:
            org = db.organizations.find_one({"_id": ObjectId(org_id)})
        except Exception:
            org = None
        if not org:
            raise HTTPException(status_code=403, detail="Organization not found")
        if org.get("status", "active") == "suspended":
            raise HTTPException(status_code=403, detail="Organization suspended")
        org_name = org.get("name")

    return {
        "id": str(user["_id"]),
        "name": user.get("name", ""),
        "email": user.get("email", ""),
        "role": role,
        "org_id": org_id,
        "org_name": org_name,
        "status": user.get("status", "active"),
        "must_change_password": bool(user.get("must_change_password", False)),
        "created_at": user.get("created_at"),
    }


def decode_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    return _load_current_user_from_id(user_id)


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Decode JWT and return current user info."""
    return decode_token(credentials.credentials)


def require_roles(*roles: str) -> Callable:
    allowed = set(roles)

    def dependency(user=Depends(get_current_user)):
        if user["role"] not in allowed:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        if user.get("must_change_password"):
            raise HTTPException(status_code=403, detail="Password change required")
        return user

    return dependency


def require_org_user(user=Depends(get_current_user)):
    if user["role"] == UserRole.SUPER_ADMIN.value:
        raise HTTPException(status_code=403, detail="Super admin cannot access clinical workspace")
    if not user.get("org_id"):
        raise HTTPException(status_code=403, detail="User is not assigned to an organization")
    if user.get("must_change_password"):
        raise HTTPException(status_code=403, detail="Password change required")
    return user


def same_org_query(user: dict) -> dict:
    return {"org_id": user["org_id"]}


async def get_websocket_user(websocket: WebSocket, token: str | None = Query(default=None)) -> dict | None:
    raw_token = token or websocket.query_params.get("token")
    if not raw_token:
        return None
    try:
        return decode_token(raw_token)
    except HTTPException:
        return None


def seed_super_admin() -> str:
    """Create or verify the startup super admin if env credentials are configured."""
    if not settings.SUPER_ADMIN_EMAIL or not settings.SUPER_ADMIN_PASSWORD:
        return "skipped"

    db = get_database()
    now = datetime.utcnow().isoformat()
    existing = db.users.find_one({"email": settings.SUPER_ADMIN_EMAIL})
    if existing:
        db.users.update_one(
            {"_id": existing["_id"]},
            {
                "$set": {
                    "role": UserRole.SUPER_ADMIN.value,
                    "org_id": None,
                    "status": "active",
                    "must_change_password": False,
                    "updated_at": now,
                }
            },
        )
        return "verified"

    db.users.insert_one(
        {
            "name": settings.SUPER_ADMIN_NAME,
            "email": settings.SUPER_ADMIN_EMAIL,
            "password_hash": hash_password(settings.SUPER_ADMIN_PASSWORD),
            "role": UserRole.SUPER_ADMIN.value,
            "org_id": None,
            "status": "active",
            "must_change_password": False,
            "created_by": None,
            "created_at": now,
            "updated_at": now,
            "last_login_at": None,
        }
    )
    return "created"


@router.post("/register")
async def register(data: UserRegister):
    """Public registration is disabled; users are provisioned by admins."""
    raise HTTPException(status_code=403, detail="Public registration is disabled")


@router.post("/login", response_model=TokenResponse)
async def login(data: UserLogin):
    """Login and receive JWT token."""
    db = get_database()

    user = db.users.find_one({"email": data.email})
    if not user or not verify_password(data.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if user.get("status", "active") == "disabled":
        raise HTTPException(status_code=403, detail="Account disabled")

    role = _normalize_role(user.get("role"))
    org_id = user.get("org_id")
    if role != UserRole.SUPER_ADMIN.value:
        if not org_id:
            raise HTTPException(status_code=403, detail="User is not assigned to an organization")
        try:
            org = db.organizations.find_one({"_id": ObjectId(org_id)})
        except Exception:
            org = None
        if not org:
            raise HTTPException(status_code=403, detail="Organization not found")
        if org.get("status", "active") == "suspended":
            raise HTTPException(status_code=403, detail="Organization suspended")

    db.users.update_one(
        {"_id": user["_id"]},
        {"$set": {"role": role, "last_login_at": datetime.utcnow().isoformat()}},
    )

    token = create_token(str(user["_id"]), role, org_id)
    user["role"] = role
    return TokenResponse(access_token=token, user=user_response(user, db))


@router.post("/change-password")
async def change_password(data: PasswordChangeRequest, user=Depends(get_current_user)):
    db = get_database()
    user_doc = db.users.find_one({"_id": ObjectId(user["id"])})
    if not user_doc or not verify_password(data.current_password, user_doc["password_hash"]):
        raise HTTPException(status_code=400, detail="Current password is incorrect")

    db.users.update_one(
        {"_id": user_doc["_id"]},
        {
            "$set": {
                "password_hash": hash_password(data.new_password),
                "must_change_password": False,
                "updated_at": datetime.utcnow().isoformat(),
            }
        },
    )
    return {"message": "Password changed"}


@router.get("/me", response_model=UserResponse)
async def get_me(user=Depends(get_current_user)):
    """Get current user profile."""
    return UserResponse(**user)
