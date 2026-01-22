from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import Dict
from app.schemas import MovieIn, MovieOut
from app.deps import get_db, get_current_user, require_roles
from app import models


router = APIRouter(prefix="/uploader", tags=["uploader"], dependencies=[Depends(require_roles("admin", "uploader"))])


@router.post("/movies", response_model=MovieOut)
def create_movie(movie_in: MovieIn, db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    if db.query(models.Movie).filter(models.Movie.title == movie_in.title).first():
        raise HTTPException(status_code=400, detail="Movie already exists")
    movie = models.Movie(title=movie_in.title,
                         description=movie_in.description or "",
                         movie_metadata=movie_in.movie_metadata or {},
                         owner=current_user)
    db.add(movie)
    db.commit()
    db.refresh(movie)
    return movie


@router.patch("/movies/{movie_id}", response_model=MovieOut)
def update_movie(movie_id: int,movie_in: MovieIn, db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    movie = db.query(models.Movie).filter(models.Movie.id == movie_id).first()
    if not movie:
        raise HTTPException(status_code=404, detail="Movie not found")
    if current_user.id != movie.owner_id and not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="You are not allowed to perform this action")

    movie.title = movie_in.title
    movie.description = movie_in.description or ""

    meta = movie_in.movie_metadata or {}
    meta.update(movie_in.movie_metadata or {})
    movie.movie_metadata = meta

    db.commit()
    db.refresh(movie)
    return movie

@router.delete("/movies/{movie_id}",status_code=204)
def delete_movie(movie_id: int, db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    movie = db.query(models.Movie).filter(models.Movie.id == movie_id).first()
    if not movie:
        raise HTTPException(status_code=404, detail="Movie not found")
    if current_user.id != movie.owner_id and not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="You are not allowed to perform this action")
    db.delete(movie)
    db.commit()
    return

@router.get("/movies/{movie_id}/stat")
def movie_stat(movie_id: int, db: Session = Depends(get_db),current_user=Depends(get_current_user)):
    movie = db.query(models.Movie).filter(models.Movie.id == movie_id).first()
    if not movie:
        raise HTTPException(status_code=404, detail="Movie not found")
    if current_user.id != movie.owner_id and not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="You are not allowed to perform this action")
    total = db.query(func.count(models.Review.id)).filter(models.Review.movie_id == movie_id).scalar() or 0
    pos = db.query(func.count(models.Review.id)).filter(models.Review.movie_id == movie_id,models.Review.sentiment_label=="positive").scalar() or 0
    nue = db.query(func.count(models.Review.id)).filter(models.Review.movie_id == movie_id,models.Review.sentiment_label=="neutral").scalar() or 0
    neg = db.query(func.count(models.Review.id)).filter(models.Review.movie_id == movie_id,models.Review.sentiment_label=="negative").scalar() or 0
    def percentage(value):
        if value <= 0:
            return 0
        return 100 * (value / total)
    return {
        "movie_id": movie_id,
        "total": total,
        "positive":{"count":pos,"percentage":percentage(pos)},
        "neutral":{"count":nue,"percentage":percentage(nue)},
        "negative":{"count":neg,"percentage":percentage(nue)},
    }



