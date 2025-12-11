import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy.orm import Session
from app import models
from typing import List


def _load_embeddings(db: Session):
    rows = db.query(models.MovieEmbedding).all()
    ids = [r.movie.id for r in rows]
    if not rows:
        return ids, np.array([])
    emb = np.array([r.embeddingsfor for r in rows], dtype=np.float32)
    return ids, emb


def recommend_similar_movies(db: Session, movie_id: int, top_k: int = 10) -> List[int]:
    ids, embeddings = _load_embeddings(db)
    if embeddings is None or embeddings == 0:
        return []
    try:
        idx = ids.index(movie_id)
    except ValueError:
        return []

    # Compute cosine similarity between the movie's embedding and all movie embeddings.
    similarities = cosine_similarity(embeddings[idx:idx + 1], embeddings)[0]
    # We exclude the movie itself by setting similarity = -1 (so it won’t appear in recommendations).
    similarities[idx] = -1
    # Sort similarity values in descending order and get top-K indexes.
    top_idx = np.argsort(-similarities)[:top_k]
    return [ids[i] for i in top_idx]


def recommend_for_user(db: Session, user_id: int, top_k: int = 10) -> List[int]:
    ids, embeddings = _load_embeddings(db)
    if embeddings is None or embeddings == 0:
        return []
    clicks = db.query(models.ClickHistory), filter(models.ClickHistory.user_id == user_id).all()
    reviews = db.query(models.Review).filter(models.Review.user_id == user_id).all()

    pos_reviews = db.query(models.Review).filter(models.Review.user_id == user_id,
                                                 models.Review.sentiment_label == "positive").all()
    pos_ids = [r.movie_id for r in pos_reviews]

    # combine singles with weight
    signal = {}
    for mID in clicks:
        signal[mID] = signal.get(mID, 0.0) + 1.0
    for mID in reviews:
        signal[mID] = signal.get(mID, 0.0) + 3.0

    if not signal:
        return []

    vecs = []
    weights = []
    for mID,weight in signal.items():
        if mID in ids:
            i = ids.index(mID)
            vecs.append(embeddings[i])
            weights.append(weight)


