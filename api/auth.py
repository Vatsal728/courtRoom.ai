from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

from pydantic import BaseModel

router = APIRouter(prefix="/auth", tags=["auth"])

SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here-change-in-prod")
ALGORITHM = "HS256"
pwd_context = CryptContext(schemes=["sha256_crypt"], deprecated="auto")

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB = os.getenv("MONGODB_DB", "courtroom_ai")
db = MongoClient(MONGODB_URI)[MONGODB_DB]

class RegisterRequest(BaseModel):
    email: str
    password: str
    name: str

class LoginRequest(BaseModel):
    email: str
    password: str

class User:
    def __init__(self, email: str, password: str, name: str):
        self.email = email
        self.hashed_password = pwd_context.hash(password)
        self.name = name
        self.created_at = datetime.now()

@router.post("/register")
def register(req: RegisterRequest):
    """Register new user"""
    if db.users.find_one({"email": req.email}):
        raise HTTPException(status_code=400, detail="Email already registered")
    
    user = User(req.email, req.password, req.name)
    result = db.users.insert_one({
        "email": user.email,
        "hashed_password": user.hashed_password,
        "name": user.name,
        "created_at": user.created_at
    })
    
    return {"user_id": str(result.inserted_id), "email": req.email}

@router.post("/login")
def login(req: LoginRequest):
    """Login and get JWT token"""
    user = db.users.find_one({"email": req.email})
    if not user or not pwd_context.verify(req.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = jwt.encode(
        {
            "sub": str(user["_id"]),
            "email": user["email"],
            "exp": datetime.utcnow() + timedelta(days=30)
        },
        SECRET_KEY,
        algorithm=ALGORITHM
    )
    
    return {
        "access_token": token,
        "token_type": "bearer",
        "user_id": str(user["_id"]),
        "name": user["name"]
    }

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
    """Verify JWT token"""
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    return user_id
