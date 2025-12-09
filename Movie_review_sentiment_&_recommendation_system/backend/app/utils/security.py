from passlib.context import CryptContext
from datetime import datetime, timedelta,timezone
import jwt
from app.config import settings

pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto")

def hash_password(password):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict):
    expires = datetime.now(timezone.utc) + timedelta(minutes=settings.AccessTokenExpirationSeconds)
    to_encode = data.copy()
    to_encode.update({"exp": expires})
    token=jwt.encode(to_encode,secret=settings.AccessTokenSecret,algorithm=settings.ALGORITHM)
    return token

def decode_access_token(token):
    try:
        payload = jwt.decode(token, secret=settings.AccessTokenSecret,algorithms=[settings.ALGORITHM])
        return payload
    except jwt.PyJWTError:
        return None