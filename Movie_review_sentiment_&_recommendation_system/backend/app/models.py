from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey, func
from sqlalchemy.orm import relationship
from app.db.base import Base
from sqlalchemy.dialects.postgresql import JSONB, ARRAY

class Product(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True,autoincrement=True)
    email = Column(String, unique=True, index=True)
    is_active = Column(Boolean,default=True)
    is_superuser = Column(Boolean,default=False)

class Movie(Base):
    __tablename__ = "movies"
    id = Column(Integer, primary_key=True, index=True,autoincrement=True)
    title = Column(String,nullable=False,index=True)
    description = Column(String,nullable=False,index=True)
    metadata = Column(JSONB,nullable=False,index=True) # genres, year, cast ....

class MovieEmbedding(Base):
    __tablename__ = "movie_embeddings"
    movie_id = Column(Integer, ForeignKey("movies.id"),primary_key=True)
    embedding = Column(JSONB,nullable=False)
    movie = relationship("Movie", backref="movie_embeddings")

class Review(Base):
    __tablename__ = "reviews"
    id = Column(Integer, primary_key=True, index=True,autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"),primary_key=True)
    movie_id = Column(Integer, ForeignKey("movies.id"),primary_key=True)
    text = Column(String,nullable=False,index=True)
    sentiment_label = Column(String,nullable=False,index=True)
    sentiment_scores = Column(JSONB,nullable=False,index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", backref="reviews")
    movie = relationship("Movie", backref="reviews")

class ClickHistory(Base):
    __tablename__ = "click_history"
    id = Column(Integer, primary_key=True, index=True,autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"),primary_key=True)
    movie_id = Column(Integer, ForeignKey("movies.id"),primary_key=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", backref="click")
    movie = relationship("Movie", backref="click")