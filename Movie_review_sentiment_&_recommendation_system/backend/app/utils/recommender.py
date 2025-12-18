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
    emb = np.array([r.embedding for r in rows], dtype=np.float32)
    return ids, emb


def recommend_similar_movies(db: Session, movie_id: int, top_k: int = 10) -> List[int]:
    ids, embeddings = _load_embeddings(db)
    if embeddings is None or len(embeddings) == 0:
        return []
    try:
        idx = ids.index(movie_id)
    except ValueError:
        return []

    # Compute cosine similarity between the movie's embedding and all movie embeddings.
    similarities = cosine_similarity(embeddings[idx:idx + 1], embeddings)[0]
    # exclude the movie itself by setting similarity = -1 (so it won’t appear in recommendations).
    similarities[idx] = -1
    # Sort similarity values in descending order and get top-K indexes.
    top_idx = np.argsort(-similarities)[:top_k]
    return [ids[i] for i in top_idx]


def recommend_for_user(db: Session, user_id: int, top_k: int = 10) -> List[int]:
    ids, embeddings = _load_embeddings(db)
    if embeddings is None or embeddings == 0:
        return []
    clicks = db.query(models.ClickHistory).filter(models.ClickHistory.user_id == user_id).all()
    clicked_ids = [c.movie.id for c in clicks]

    pos_reviews = db.query(models.Review).filter(models.Review.user_id == user_id,
                                                 models.Review.sentiment_label == "positive").all()
    pos_ids = [r.movie_id for r in pos_reviews]

    # combine singles with weight
    signal = {}
    for mID in clicked_ids:
        signal[mID] = signal.get(mID, 0.0) + 1.0
    for mID in pos_ids:
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
    if not vecs:
        return []

    vecs = np.array(vecs)
    weights = np.array(weights).reshape(-1,1)
    agg = (vecs * weights).sum(axis=0)/weights.sum()
    sims = cosine_similarity(agg.reshape(1,-1), embeddings)[0]

    # exclude already interacted
    excluded = set(clicked_ids) | set([r.movie.id for r in db.query(models.Review).filter(models.Review.user_id == user_id).all()])
    for ex in excluded:
        if ex in ids:
            sims[ids.index(ex)] = -1

    top_idx = np.argsort(-sims)[:top_k]
    return [ids[i] for i in top_idx]