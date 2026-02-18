"""
Authentication Routes — Register, Login, Me
"""
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from datetime import datetime, timedelta
from jose import jwt, JWTError
from passlib.context import CryptContext
from bson import ObjectId
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings
from database import get_database
from models.schemas import UserRegister, UserLogin, UserResponse, TokenResponse

router = APIRouter(prefix="/api/auth", tags=["Authentication"])
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()


def create_token(user_id: str, role: str) -> str:
    """Create JWT token."""
    payload = {
        "sub": user_id,
        "role": role,
        "exp": datetime.utcnow() + timedelta(hours=settings.JWT_EXPIRATION_HOURS)
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Decode JWT and return user info."""
    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.JWT_SECRET,
            algorithms=[settings.JWT_ALGORITHM]
        )
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        db = get_database()
        user = db.users.find_one({"_id": ObjectId(user_id)})
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        
        return {
            "id": str(user["_id"]),
            "name": user["name"],
            "email": user["email"],
            "role": user["role"]
        }
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


@router.post("/register", response_model=TokenResponse)
async def register(data: UserRegister):
    """Register a new user."""
    db = get_database()
    
    # Check if email exists
    if db.users.find_one({"email": data.email}):
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create user
    user_doc = {
        "name": data.name,
        "email": data.email,
        "password_hash": pwd_context.hash(data.password),
        "role": data.role.value,
        "created_at": datetime.utcnow().isoformat()
    }
    
    result = db.users.insert_one(user_doc)
    user_id = str(result.inserted_id)
    
    token = create_token(user_id, data.role.value)
    
    return TokenResponse(
        access_token=token,
        user=UserResponse(
            id=user_id,
            name=data.name,
            email=data.email,
            role=data.role,
            created_at=user_doc["created_at"]
        )
    )


@router.post("/login", response_model=TokenResponse)
async def login(data: UserLogin):
    """Login and receive JWT token."""
    db = get_database()
    
    user = db.users.find_one({"email": data.email})
    if not user or not pwd_context.verify(data.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    user_id = str(user["_id"])
    token = create_token(user_id, user["role"])
    
    return TokenResponse(
        access_token=token,
        user=UserResponse(
            id=user_id,
            name=user["name"],
            email=user["email"],
            role=user["role"],
            created_at=user.get("created_at")
        )
    )


@router.get("/me", response_model=UserResponse)
async def get_me(user=Depends(get_current_user)):
    """Get current user profile."""
    return UserResponse(**user)
