from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # DATABASE link
    DATABASE_URL: str = "postgresql://postgres:1234@localhost:5432/moviedb"
    # SECRET_KEY (https://generate-random.org/salts)
    SECRET_KEY: str = "fVyEQqegn5hfbVrG6TItwtwp8DDo3tF5"
    ACCESS_TOKEN_EXPIRE_MINUTES : int = 60*24*7
    ALGORITHM:str  = "HS256"

    # MY model paths
    SENTIMENTAL_MODEL_PATH: str = ""
    EMBEDDING_MODEL_PATH: str = ""

settings = Settings()