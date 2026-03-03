from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.utils.auth_utils import hash_password, verify_password, create_access_token
from backend.utils.database import users_collection

router = APIRouter(prefix="/auth", tags=["Authentication"])


class UserRegister(BaseModel):
    email: str
    password: str


class UserLogin(BaseModel):
    email: str
    password: str


@router.post("/register")
def register(user: UserRegister):
    # Check if user already exists
    existing_user = users_collection.find_one({"email": user.email})

    if existing_user:
        raise HTTPException(status_code=400, detail="User already exists")

    # Insert new user
    users_collection.insert_one({
        "email": user.email,
        "password": hash_password(user.password)
    })

    return {"message": "User registered successfully"}


@router.post("/login")
def login(user: UserLogin):
    # Find user in database
    db_user = users_collection.find_one({"email": user.email})

    if not db_user:
        raise HTTPException(status_code=400, detail="User not found")

    # Verify password
    if not verify_password(user.password, db_user["password"]):
        raise HTTPException(status_code=400, detail="Incorrect password")

    # Create JWT token
    access_token = create_access_token({"sub": user.email})

    return {
        "access_token": access_token,
        "token_type": "bearer"
    }