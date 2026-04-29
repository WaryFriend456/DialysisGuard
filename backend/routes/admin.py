"""
Platform administration routes for super admins.
"""
from datetime import datetime
import os
import sys

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException
from pymongo.errors import DuplicateKeyError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import get_database
from models.schemas import (
    AdminUserCreate,
    CreatedUserResponse,
    OrganizationCreate,
    OrganizationResponse,
    OrganizationUpdate,
    UserRole,
)
from routes.auth import generate_temporary_password, hash_password, require_roles, user_response

router = APIRouter(prefix="/api/admin", tags=["Platform Admin"])


def _object_id(value: str) -> ObjectId:
    try:
        return ObjectId(value)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid ID")


def _clean_code(code: str) -> str:
    return code.strip().upper().replace(" ", "_")


def _organization_response(org: dict, db) -> OrganizationResponse:
    org_id = str(org["_id"])
    return OrganizationResponse(
        id=org_id,
        name=org.get("name", ""),
        code=org.get("code", ""),
        status=org.get("status", "active"),
        address=org.get("address"),
        phone=org.get("phone"),
        email=org.get("email"),
        staff_count=db.users.count_documents({"org_id": org_id}),
        patient_count=db.patients.count_documents({"org_id": org_id}),
        created_by=org.get("created_by"),
        created_at=org.get("created_at"),
        updated_at=org.get("updated_at"),
    )


@router.get("/organizations")
async def list_organizations(user=Depends(require_roles(UserRole.SUPER_ADMIN.value))):
    db = get_database()
    orgs = list(db.organizations.find({}).sort("created_at", -1))
    return {"organizations": [_organization_response(org, db) for org in orgs]}


@router.post("/organizations", response_model=OrganizationResponse)
async def create_organization(
    data: OrganizationCreate,
    user=Depends(require_roles(UserRole.SUPER_ADMIN.value)),
):
    db = get_database()
    now = datetime.utcnow().isoformat()
    doc = {
        "name": data.name,
        "code": _clean_code(data.code),
        "status": "active",
        "address": data.address,
        "phone": data.phone,
        "email": str(data.email) if data.email else None,
        "created_by": user["id"],
        "created_at": now,
        "updated_at": now,
    }
    try:
        result = db.organizations.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(status_code=400, detail="Organization code already exists")
    doc["_id"] = result.inserted_id
    return _organization_response(doc, db)


@router.get("/organizations/{org_id}", response_model=OrganizationResponse)
async def get_organization(org_id: str, user=Depends(require_roles(UserRole.SUPER_ADMIN.value))):
    db = get_database()
    org = db.organizations.find_one({"_id": _object_id(org_id)})
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")
    return _organization_response(org, db)


@router.put("/organizations/{org_id}", response_model=OrganizationResponse)
async def update_organization(
    org_id: str,
    data: OrganizationUpdate,
    user=Depends(require_roles(UserRole.SUPER_ADMIN.value)),
):
    db = get_database()
    updates = {k: v for k, v in data.model_dump().items() if v is not None}
    if "code" in updates:
        updates["code"] = _clean_code(updates["code"])
    if "email" in updates and updates["email"] is not None:
        updates["email"] = str(updates["email"])
    updates["updated_at"] = datetime.utcnow().isoformat()
    try:
        result = db.organizations.update_one({"_id": _object_id(org_id)}, {"$set": updates})
    except DuplicateKeyError:
        raise HTTPException(status_code=400, detail="Organization code already exists")
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Organization not found")
    return await get_organization(org_id, user)


@router.post("/organizations/{org_id}/suspend", response_model=OrganizationResponse)
async def suspend_organization(org_id: str, user=Depends(require_roles(UserRole.SUPER_ADMIN.value))):
    db = get_database()
    result = db.organizations.update_one(
        {"_id": _object_id(org_id)},
        {"$set": {"status": "suspended", "updated_at": datetime.utcnow().isoformat()}},
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Organization not found")
    return await get_organization(org_id, user)


@router.post("/organizations/{org_id}/activate", response_model=OrganizationResponse)
async def activate_organization(org_id: str, user=Depends(require_roles(UserRole.SUPER_ADMIN.value))):
    db = get_database()
    result = db.organizations.update_one(
        {"_id": _object_id(org_id)},
        {"$set": {"status": "active", "updated_at": datetime.utcnow().isoformat()}},
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Organization not found")
    return await get_organization(org_id, user)


@router.get("/organizations/{org_id}/users")
async def list_organization_users(org_id: str, user=Depends(require_roles(UserRole.SUPER_ADMIN.value))):
    db = get_database()
    if not db.organizations.find_one({"_id": _object_id(org_id)}):
        raise HTTPException(status_code=404, detail="Organization not found")
    users = list(db.users.find({"org_id": org_id}).sort("created_at", -1))
    return {"users": [user_response(item, db) for item in users]}


@router.post("/organizations/{org_id}/org-admins", response_model=CreatedUserResponse)
async def create_org_admin(
    org_id: str,
    data: AdminUserCreate,
    user=Depends(require_roles(UserRole.SUPER_ADMIN.value)),
):
    db = get_database()
    if not db.organizations.find_one({"_id": _object_id(org_id)}):
        raise HTTPException(status_code=404, detail="Organization not found")
    if db.users.find_one({"email": str(data.email)}):
        raise HTTPException(status_code=400, detail="Email already registered")

    temp_password = generate_temporary_password()
    now = datetime.utcnow().isoformat()
    doc = {
        "name": data.name,
        "email": str(data.email),
        "password_hash": hash_password(temp_password),
        "role": UserRole.ORG_ADMIN.value,
        "org_id": org_id,
        "status": "active",
        "must_change_password": True,
        "created_by": user["id"],
        "created_at": now,
        "updated_at": now,
        "last_login_at": None,
    }
    result = db.users.insert_one(doc)
    doc["_id"] = result.inserted_id
    return CreatedUserResponse(user=user_response(doc, db), temporary_password=temp_password)


@router.post("/users/{user_id}/disable")
async def disable_user(user_id: str, user=Depends(require_roles(UserRole.SUPER_ADMIN.value))):
    db = get_database()
    result = db.users.update_one(
        {"_id": _object_id(user_id), "role": {"$ne": UserRole.SUPER_ADMIN.value}},
        {"$set": {"status": "disabled", "updated_at": datetime.utcnow().isoformat()}},
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="User not found")
    return {"message": "User disabled"}


@router.post("/users/{user_id}/activate")
async def activate_user(user_id: str, user=Depends(require_roles(UserRole.SUPER_ADMIN.value))):
    db = get_database()
    result = db.users.update_one(
        {"_id": _object_id(user_id), "role": {"$ne": UserRole.SUPER_ADMIN.value}},
        {"$set": {"status": "active", "updated_at": datetime.utcnow().isoformat()}},
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="User not found")
    return {"message": "User activated"}


@router.post("/users/{user_id}/reset-password", response_model=CreatedUserResponse)
async def reset_user_password(user_id: str, user=Depends(require_roles(UserRole.SUPER_ADMIN.value))):
    db = get_database()
    target = db.users.find_one({"_id": _object_id(user_id), "role": {"$ne": UserRole.SUPER_ADMIN.value}})
    if not target:
        raise HTTPException(status_code=404, detail="User not found")
    temp_password = generate_temporary_password()
    db.users.update_one(
        {"_id": target["_id"]},
        {
            "$set": {
                "password_hash": hash_password(temp_password),
                "must_change_password": True,
                "updated_at": datetime.utcnow().isoformat(),
            }
        },
    )
    target["password_hash"] = ""
    target["must_change_password"] = True
    return CreatedUserResponse(user=user_response(target, db), temporary_password=temp_password)
