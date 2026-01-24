from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.deps import get_current_user, get_db
from app import models
from app.schemas import ReviewCreate, ReviewOut, ClickOut, RecommendForUserRequest, MovieOut
from app.utils.recommender import  recommend_similar_movies # ,recommend_for_user,
from app.utils.security import create_access_token
from app.scripts.predict import predict_sentiment

router = APIRouter(prefix="/interact", tags=["interact"])


def predict_sentiment_dummy(text: str):
    label_id, confidence = predict_sentiment(text)
    if label_id == 1:
        return {
            "label": "positive",
            "score": {
                "positive": confidence,
                "negative": 1-confidence,
            }
        }
    return {
        "label": "negative",
        "score": {
            "positive": 1-confidence,
            "negative": confidence,
        }
    }

@router.post("/movies/{movie_id}/click", response_model=ClickOut)
def save_click(movie_id: int, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    movie = db.get(models.Movie, movie_id)
    if not movie:
        raise HTTPException(status_code=404, detail="Movie not found")
    click = models.ClickHistory(user_id=current_user.id, movie_id=movie.id)
    db.add(click)
    db.commit()
    db.refresh(click)
    return click

@router.post("/movies/{movie_id}/review", response_model=ReviewOut)
def save_review(movie_id: int, payload: ReviewCreate,db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    movie = db.get(models.Movie, movie_id)
    if not movie:
        raise HTTPException(status_code=404, detail="Movie not found")
    pred = predict_sentiment_dummy(payload.text)
    review = models.Review(
        user_id=current_user.id,
        movie_id=movie.id,
        text=payload.text,
        sentiment_label=pred["label"],
        sentiment_scores=pred["score"]
    )
    db.add(review)
    db.commit()
    db.refresh(review)
    return review

# @router.post("/recommend_for_user", response_model=List[MovieOut])
# def recommend_for_user_endpoint(request: RecommendForUserRequest, db: Session = Depends(get_db)):
#     rec_ids = recommend