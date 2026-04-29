"""
Patient CRUD Routes
"""
from fastapi import APIRouter, HTTPException, Depends, Query
from bson import ObjectId
from datetime import datetime
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import get_database
from models.schemas import PatientCreate, PatientUpdate, PatientResponse, UserRole
from routes.auth import require_org_user

router = APIRouter(prefix="/api/patients", tags=["Patients"])


def _to_response(doc: dict) -> PatientResponse:
    """Convert MongoDB document to PatientResponse."""
    return PatientResponse(
        id=str(doc["_id"]),
        org_id=doc.get("org_id"),
        name=doc.get("name"),
        age=doc["age"],
        gender=doc["gender"],
        weight=doc["weight"],
        diabetes=doc.get("diabetes", False),
        hypertension=doc.get("hypertension", False),
        kidney_failure_cause=doc.get("kidney_failure_cause", "Other"),
        creatinine=doc.get("creatinine", 5.0),
        urea=doc.get("urea", 50.0),
        potassium=doc.get("potassium", 4.5),
        hemoglobin=doc.get("hemoglobin", 11.0),
        hematocrit=doc.get("hematocrit", 33.0),
        albumin=doc.get("albumin", 3.8),
        dialysis_duration=doc.get("dialysis_duration", 4.0),
        dialysis_frequency=doc.get("dialysis_frequency", 3),
        dialysate_composition=doc.get("dialysate_composition", "Standard"),
        vascular_access_type=doc.get("vascular_access_type", "Fistula"),
        dialyzer_type=doc.get("dialyzer_type", "High-flux"),
        urine_output=doc.get("urine_output", 500),
        dry_weight=doc.get("dry_weight", 70.0),
        fluid_removal_rate=doc.get("fluid_removal_rate", 350),
        disease_severity=doc.get("disease_severity", "Moderate"),
        created_by=doc.get("created_by"),
        created_at=doc.get("created_at")
    )


@router.get("/")
async def list_patients(
    search: str = Query(None),
    limit: int = Query(50, ge=1, le=100),
    skip: int = Query(0, ge=0),
    user=Depends(require_org_user)
):
    """List patients with optional search."""
    db = get_database()
    query = {"org_id": user["org_id"]}
    
    if search:
        query["$or"] = [
            {"name": {"$regex": search, "$options": "i"}},
            {"gender": {"$regex": search, "$options": "i"}},
        ]
    
    patients = list(db.patients.find(query).sort("created_at", -1).skip(skip).limit(limit))
    total = db.patients.count_documents(query)
    
    return {
        "patients": [_to_response(p) for p in patients],
        "total": total
    }


@router.post("/", response_model=PatientResponse)
async def create_patient(data: PatientCreate, user=Depends(require_org_user)):
    """Create a new patient."""
    if user["role"] not in {UserRole.DOCTOR.value, UserRole.ORG_ADMIN.value}:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    db = get_database()
    
    doc = data.model_dump()
    doc["org_id"] = user["org_id"]
    doc["created_by"] = user["id"]
    doc["created_at"] = datetime.utcnow().isoformat()
    doc["updated_at"] = doc["created_at"]
    
    result = db.patients.insert_one(doc)
    doc["_id"] = result.inserted_id
    
    return _to_response(doc)


@router.get("/{patient_id}", response_model=PatientResponse)
async def get_patient(patient_id: str, user=Depends(require_org_user)):
    """Get patient by ID."""
    db = get_database()
    
    try:
        patient = db.patients.find_one({"_id": ObjectId(patient_id), "org_id": user["org_id"]})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid patient ID")
    
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    return _to_response(patient)


@router.put("/{patient_id}", response_model=PatientResponse)
async def update_patient(patient_id: str, data: PatientUpdate, user=Depends(require_org_user)):
    """Update patient data."""
    if user["role"] not in {UserRole.DOCTOR.value, UserRole.ORG_ADMIN.value}:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    db = get_database()
    
    updates = {k: v for k, v in data.model_dump().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")
    updates["updated_at"] = datetime.utcnow().isoformat()
    
    try:
        result = db.patients.update_one(
            {"_id": ObjectId(patient_id), "org_id": user["org_id"]},
            {"$set": updates}
        )
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid patient ID")
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    return await get_patient(patient_id, user)


@router.delete("/{patient_id}")
async def delete_patient(patient_id: str, user=Depends(require_org_user)):
    """Delete a patient."""
    if user["role"] not in {UserRole.DOCTOR.value, UserRole.ORG_ADMIN.value}:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    db = get_database()
    
    try:
        result = db.patients.delete_one({"_id": ObjectId(patient_id), "org_id": user["org_id"]})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid patient ID")
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    return {"message": "Patient deleted"}
