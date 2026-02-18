"""
MongoDB Database Connection
"""
from pymongo import MongoClient
from config import settings

client = None
db = None


def get_database():
    """Get the MongoDB database instance."""
    global client, db
    if db is None:
        client = MongoClient(settings.MONGODB_URI)
        db = client[settings.MONGODB_DB]
        _create_indexes()
    return db


def _create_indexes():
    """Create database indexes for performance."""
    db.users.create_index("email", unique=True)
    db.patients.create_index("created_by")
    db.sessions.create_index("patient_id")
    db.sessions.create_index("status")
    db.alerts.create_index([("session_id", 1), ("created_at", -1)])
    db.alerts.create_index("patient_id")
    db.alerts.create_index("acknowledged")


def close_database():
    """Close the MongoDB connection."""
    global client, db
    if client:
        client.close()
        client = None
        db = None
