from sentence_transformers import SentenceTransformer
from app.db.session import SessionLocal
from app import models

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def embed_text(text,model):
    return model.encode(text,show_progress_bar=False).tolist()

def main():
    model = SentenceTransformer(MODEL_NAME)
    db = SessionLocal()
    movies = db.query(models.Movie).all()
    for movie in movies:
        text = (movie.text or "") + " " + (movie.description or "") + " " + (movie.year or "")
        embeddings = model.encode(text).tolist()
        row = db.query(models.MovieEmbedding).filter(models.MovieEmbedding.movie_id == movie.movie_id).first()
        if row :
            row.embedding = embeddings
        else:
            row = models.MovieEmbedding(movie_id=movie.movie_id, embedding=embeddings)
            db.add(row)
    db.commit()
    db.close()

if __name__ == "__main__":
    main()