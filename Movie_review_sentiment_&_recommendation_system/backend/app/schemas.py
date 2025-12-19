from datetime import datetime
from pydantic import BaseModel, EmailStr
from typing import Optional, List, Any, Dict


# --- users ---
class UserCreate(BaseModel):
    email: EmailStr
    password: str
    username: str
    full_name: Optional[str] = None
    phone: Optional[str] = None
    role: Optional[str] = "user"


class UserOut(BaseModel):
    id: int
    email: EmailStr
    username: str
    full_name: Optional[str] = None
    phone: Optional[str] = None
    role: Optional[str] = "user"
    is_active: bool = True
    is_superuser: bool = True

    class Config:
        from_attributes = True


class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    phone: Optional[str] = None
    role: Optional[str] = "user"
    is_active: Optional[bool] = True


# --- movie ---
class MovieIn(BaseModel):
    title: str
    description: Optional[str] = None
    movie_metadata: Optional[Dict] = None


class MovieOut(BaseModel):
    id: int
    title: str
    description: Optional[str] = None
    movie_metadata: Optional[Dict] = None

    class Config:
        from_attributes = True

# --- reviews ---
class ReviewCreate(BaseModel):
    text: str

class ReviewOut(BaseModel):
    id: int
    user_id: int
    movie_id: int
    text: str
    sentiment_label: str
    sentiment_scores: Dict[str, float]
    created_at: datetime
    class Config:
        from_attributes = True

# --- click history ---
class ClickOut(BaseModel):
    id: int
    user_id: int
    movie_id: int
    created_at: datetime
    class Config:
        from_attributes = True

# --- recommendation requests ---
class RecommendationCreate(BaseModel):
    movie_id: int
    top_k: int = 10

class RecommendForUserRequest(BaseModel):
    top_k: int = 10