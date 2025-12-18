# app/main.py
from fastapi import FastAPI
from app.db.base import Base
from app.db.session import engine
from app.routers import auth_router, admin_router, uploader_router, interact_router

# create tables (development only). In production use Alembic.
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Movie Recommender+ Sentiment API")

app.include_router(auth_router.router)
app.include_router(admin_router.router)