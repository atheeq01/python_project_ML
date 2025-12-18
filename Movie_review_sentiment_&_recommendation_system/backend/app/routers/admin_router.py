from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from app.deps import get_db, require_roles
from app import models
from app.schemas import UserOut, MovieOut, ReviewOut

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_roles("admin"))])


# ---------user things --------------

@router.get("/users", response_model=List[UserOut])
def list_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return db.query(models.User).offset(skip).limit(limit).all()


@router.patch("/user/{user_id}/roles", response_model=UserOut)
def change_the_role(user_id: int, new_role: str, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.role = new_role
    db.commit()
    db.refresh(user)
    return user


@router.delete("user/{user_id}", status_code=204)
def delete_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    db.delete(user)
    db.commit()
    return


# ---------Movie things --------------
@router.get("/movies", response_model=List[MovieOut])
def list_movies(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return db.query(models.Movie).offset(skip).limit(limit).all()


@router.delete("/movie/{movie_id}", status_code=204)
def delete_movie(movie_id: int, db: Session = Depends(get_db)):
    movie = db.query(models.Movie).filter(models.Movie.id == movie_id).first()
    if not movie:
        raise HTTPException(status_code=404, detail="Movie not found")
    db.delete(movie)
    db.commit()
    return


@router.get("/reviews", response_model=List[ReviewOut])
def list_reviews(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return  db.query(models.Review).offset(skip).limit(limit).all()

@router.delete("/review/{review_id}",status_code=204)
def delete_review(review_id: int, db: Session = Depends(get_db)):
    review = db.query(models.Review).filter(models.Movie.id == review_id).first()
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")
    db.delete(review)
    db.commit()
    return


