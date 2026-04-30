"""
Hospital administration routes for organization admins.
"""
from datetime import datetime
import os
import sys

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException
from pymongo.errors import DuplicateKeyError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import get_database
from models.schemas import AdminUserCreate, AdminUserUpdate, CreatedUserResponse, UserRole
from routes.auth import generate_temporary_password, hash_password, require_roles, user_response

router = APIRouter(prefix="/api/org", tags=["Organization Admin"])


def _object_id(value: str) -> ObjectId:
    try:
        return ObjectId(value)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid ID")


@router.get("/summary")
async def org_summary(user=Depends(require_roles(UserRole.ORG_ADMIN.value))):
    db = get_database()
    org_id = user["org_id"]
    org = db.organizations.find_one({"_id": _object_id(org_id)})
    return {
        "organization": {
            "id": org_id,
            "name": org.get("name") if org else user.get("org_name"),
            "status": org.get("status", "active") if org else "active",
        },
        "staff_count": db.users.count_documents({"org_id": org_id}),
        "doctor_count": db.users.count_documents({"org_id": org_id, "role": UserRole.DOCTOR.value}),
        "nurse_count": db.users.count_documents({"org_id": org_id, "role": UserRole.NURSE.value}),
        "patient_count": db.patients.count_documents({"org_id": org_id}),
    }


@router.get("/staff")
async def list_staff(user=Depends(require_roles(UserRole.ORG_ADMIN.value))):
    db = get_database()
    users = list(
        db.users.find({"org_id": user["org_id"], "role": {"$in": [UserRole.DOCTOR.value, UserRole.NURSE.value]}})
        .sort("created_at", -1)
    )
    return {"users": [user_response(item, db) for item in users]}


@router.post("/staff", response_model=CreatedUserResponse)
async def create_staff(data: AdminUserCreate, user=Depends(require_roles(UserRole.ORG_ADMIN.value))):
    if data.role not in {UserRole.DOCTOR, UserRole.NURSE}:
        raise HTTPException(status_code=400, detail="Staff role must be doctor or nurse")
    db = get_database()
    if db.users.find_one({"email": str(data.email)}):
        raise HTTPException(status_code=400, detail="Email already registered")

    temp_password = generate_temporary_password()
    now = datetime.utcnow().isoformat()
    doc = {
        "name": data.name,
        "email": str(data.email),
        "password_hash": hash_password(temp_password),
        "role": data.role.value,
        "org_id": user["org_id"],
        "status": "active",
        "must_change_password": True,
        "created_by": user["id"],
        "created_at": now,
        "updated_at": now,
        "last_login_at": None,
    }
    try:
        result = db.users.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(status_code=400, detail="Email already registered")
    doc["_id"] = result.inserted_id
    return CreatedUserResponse(user=user_response(doc, db), temporary_password=temp_password)


@router.put("/staff/{staff_id}")
async def update_staff(
    staff_id: str,
    data: AdminUserUpdate,
    user=Depends(require_roles(UserRole.ORG_ADMIN.value)),
):
    updates = {k: v for k, v in data.model_dump().items() if v is not None}
    if "role" in updates:
        role = updates["role"]
        if role not in {UserRole.DOCTOR, UserRole.NURSE}:
            raise HTTPException(status_code=400, detail="Staff role must be doctor or nurse")
        updates["role"] = role.value
    if "email" in updates:
        updates["email"] = str(updates["email"])
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")
    updates["updated_at"] = datetime.utcnow().isoformat()

    db = get_database()
    try:
        result = db.users.update_one(
            {
                "_id": _object_id(staff_id),
                "org_id": user["org_id"],
                "role": {"$in": [UserRole.DOCTOR.value, UserRole.NURSE.value]},
            },
            {"$set": updates},
        )
    except DuplicateKeyError:
        raise HTTPException(status_code=400, detail="Email already registered")
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Staff user not found")
    target = db.users.find_one({"_id": _object_id(staff_id), "org_id": user["org_id"]})
    return user_response(target, db)


@router.post("/staff/{staff_id}/disable")
async def disable_staff(staff_id: str, user=Depends(require_roles(UserRole.ORG_ADMIN.value))):
    db = get_database()
    result = db.users.update_one(
        {
            "_id": _object_id(staff_id),
            "org_id": user["org_id"],
            "role": {"$in": [UserRole.DOCTOR.value, UserRole.NURSE.value]},
        },
        {"$set": {"status": "disabled", "updated_at": datetime.utcnow().isoformat()}},
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Staff user not found")
    return {"message": "Staff user disabled"}


@router.post("/staff/{staff_id}/activate")
async def activate_staff(staff_id: str, user=Depends(require_roles(UserRole.ORG_ADMIN.value))):
    db = get_database()
    result = db.users.update_one(
        {
            "_id": _object_id(staff_id),
            "org_id": user["org_id"],
            "role": {"$in": [UserRole.DOCTOR.value, UserRole.NURSE.value]},
        },
        {"$set": {"status": "active", "updated_at": datetime.utcnow().isoformat()}},
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Staff user not found")
    return {"message": "Staff user activated"}


@router.post("/staff/{staff_id}/reset-password", response_model=CreatedUserResponse)
async def reset_staff_password(staff_id: str, user=Depends(require_roles(UserRole.ORG_ADMIN.value))):
    db = get_database()
    target = db.users.find_one(
        {
            "_id": _object_id(staff_id),
            "org_id": user["org_id"],
            "role": {"$in": [UserRole.DOCTOR.value, UserRole.NURSE.value]},
        }
    )
    if not target:
        raise HTTPException(status_code=404, detail="Staff user not found")

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
    target["must_change_password"] = True
    return CreatedUserResponse(user=user_response(target, db), temporary_password=temp_password)
