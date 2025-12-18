from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import or_
from fastapi.security import OAuth2PasswordRequestForm
from app.schemas import UserCreate, UserOut
from app.db.session import SessionLocal
from app import models
from app.utils.security import hash_password, verify_password, create_access_token
from app.deps import get_db, require_roles

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserOut)
def register(user_in: UserCreate, db: Session = Depends(get_db)):
    if db.query(models.User).filter(models.User.email == user_in.email).first():
        raise HTTPException(status_code=400, detail="Email already registered.")
    if db.query(models.User).filter(models.User.username == user_in.username).first():
        raise HTTPException(status_code=400, detail="Username already taken")
    user = models.User(
        email=user_in.email,
        username=user_in.username,
        full_name=user_in.full_name,
        phone=user_in.phone,
        role="user",
        hashed_password=hash_password(user_in.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.post("/create_user_with_role", response_model=UserOut, dependencies=[Depends(require_roles("admin"))])
def create_user_with_role(user_in: UserCreate, db: Session = Depends(get_db)):
    if db.query(models.User).filter(models.User.email == user_in.email).first():
        raise HTTPException(status_code=400, detail="Email already registered.")
    if db.query(models.User).filter(models.User.username == user_in.username).first():
        raise HTTPException(status_code=400, detail="Username already taken")
    user = models.User(
        email=user_in.email,
        username=user_in.username,
        full_name=user_in.full_name,
        phone=user_in.phone,
        role= user_in.role or "user",
        hashed_password=hash_password(user_in.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.post("/token")
def login(from_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(models.User).filter(
        or_(
            models.User.email == from_data.username,
            models.User.username == from_data.username
        )
    ).first()
    if not user or not verify_password(from_data.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Credentials")
    token = create_access_token({"sub": user.email, "id": user.id})
    return {"access_token": token, "token_type": "bearer"}
