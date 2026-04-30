"""
One-time migration to assign legacy data to a default organization.

Run from backend directory:
    python scripts/migrate_orgs.py
"""
from datetime import datetime
import os
import sys

from bson import ObjectId

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import get_database


def main():
    db = get_database()
    now = datetime.utcnow().isoformat()

    org = db.organizations.find_one({"code": "DEFAULT"})
    if not org:
        result = db.organizations.insert_one(
            {
                "name": "Default Hospital",
                "code": "DEFAULT",
                "status": "active",
                "address": None,
                "phone": None,
                "email": None,
                "created_by": None,
                "created_at": now,
                "updated_at": now,
            }
        )
        org_id = str(result.inserted_id)
        print(f"Created Default Hospital: {org_id}")
    else:
        org_id = str(org["_id"])
        print(f"Using existing Default Hospital: {org_id}")

    users_result = db.users.update_many(
        {"role": {"$ne": "super_admin"}},
        {
            "$set": {
                "org_id": org_id,
                "status": "active",
                "must_change_password": False,
                "updated_at": now,
            }
        },
    )
    caregiver_result = db.users.update_many({"role": "caregiver"}, {"$set": {"role": "nurse"}})

    patients_result = db.patients.update_many(
        {"org_id": {"$exists": False}},
        {"$set": {"org_id": org_id, "updated_at": now}},
    )

    session_updates = 0
    for session in db.sessions.find({"org_id": {"$exists": False}}):
        target_org_id = org_id
        patient_id = session.get("patient_id")
        if patient_id:
            try:
                patient = db.patients.find_one({"_id": ObjectId(patient_id)})
            except Exception:
                patient = None
            if patient and patient.get("org_id"):
                target_org_id = patient["org_id"]
        db.sessions.update_one({"_id": session["_id"]}, {"$set": {"org_id": target_org_id}})
        session_updates += 1

    alert_updates = 0
    for alert in db.alerts.find({"org_id": {"$exists": False}}):
        target_org_id = org_id
        patient_id = alert.get("patient_id")
        session_id = alert.get("session_id")
        patient = None
        if patient_id:
            try:
                patient = db.patients.find_one({"_id": ObjectId(patient_id)})
            except Exception:
                patient = None
        if patient and patient.get("org_id"):
            target_org_id = patient["org_id"]
        elif session_id:
            try:
                session = db.sessions.find_one({"_id": ObjectId(session_id)})
            except Exception:
                session = None
            if session and session.get("org_id"):
                target_org_id = session["org_id"]
        db.alerts.update_one({"_id": alert["_id"]}, {"$set": {"org_id": target_org_id}})
        alert_updates += 1

    print(f"Users assigned: {users_result.modified_count}")
    print(f"Caregivers renamed to nurses: {caregiver_result.modified_count}")
    print(f"Patients assigned: {patients_result.modified_count}")
    print(f"Sessions assigned: {session_updates}")
    print(f"Alerts assigned: {alert_updates}")


if __name__ == "__main__":
    main()
