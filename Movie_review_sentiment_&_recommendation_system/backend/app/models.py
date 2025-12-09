from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey, func
from sqlalchemy.orm import relationship
from app.db.base import Base
from sqlalchemy.dialects.postgresql import JSONB


class User(Base):
    __tablename__ = "users"
    # keys
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    # attributes
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String, nullable=True)
    phone = Column(String, nullable=True)
    role = Column(String, default="user")  # admin,uploader, user
    hashed_password = Column(String, nullable=False)

    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    # Relationships
    movies = relationship("Movie", back_populates="owner")
    reviews = relationship("Review", back_populates="user")
    clicks = relationship("ClickHistory", back_populates="user")


class Movie(Base):
    __tablename__ = "movies"
    # keys
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    owner_id = Column(Integer, ForeignKey("users.id"))
    # attribute
    title = Column(String, nullable=False, index=True)
    description = Column(String, nullable=False)
    metadata = Column(JSONB, nullable=False)  # genres, year, cast ....
    # relationship
    owner = relationship("User", back_populates="movies")
    reviews = relationship("Review", back_populates="movie")
    clicks = relationship("ClickHistory", back_populates="movie")


class MovieEmbedding(Base):
    __tablename__ = "movie_embeddings"

    movie_id = Column(Integer, ForeignKey("movies.id"), primary_key=True)
    embedding = Column(JSONB, nullable=False)

    movie = relationship("Movie", backref="movie_embeddings")


class Review(Base):
    __tablename__ = "reviews"
    # keys
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    movie_id = Column(Integer, ForeignKey("movies.id"))
    # attributes
    text = Column(String, nullable=False, index=True)
    sentiment_label = Column(String, nullable=False)
    sentiment_scores = Column(JSONB, nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    # Relationships
    movie = relationship("Movie", back_populates="reviews")
    user = relationship("User", back_populates="reviews")


class ClickHistory(Base):
    __tablename__ = "click_history"
    # keys
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    movie_id = Column(Integer, ForeignKey("movies.id"))

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    # relationships
    user = relationship("User", backref="click_history")
    movie = relationship("Movie", backref="click_history")
