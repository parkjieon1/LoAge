from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr
from passlib.hash import bcrypt
from datetime import datetime

router = APIRouter()

# In-memory user store to demonstrate auth flow without a DB
fake_users: dict[str, dict] = {}

class SignUpIn(BaseModel):
    nickname: str
    email: EmailStr
    password: str

class LoginIn(BaseModel):
    email: EmailStr
    password: str

@router.post("/signup")
def signup(payload: SignUpIn):
    if payload.email in fake_users:
        raise HTTPException(status_code=400, detail="Email already exists.")
    password_hash = bcrypt.hash(payload.password)
    fake_users[payload.email] = {
        "nickname": payload.nickname,
        "password_hash": password_hash,
        "created_at": datetime.utcnow(),
    }
    return {"ok": True}

@router.post("/login")
def login(payload: LoginIn):
    user = fake_users.get(payload.email)
    if not user or not bcrypt.verify(payload.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    return {"access_token": "dummy-session-token", "token_type": "bearer"}
