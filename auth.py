"""Authentication utilities: password hashing, session tokens, CSRF tokens, and rate limiting."""
import hmac
import hashlib
import time
import secrets
import bcrypt
from collections import defaultdict
from typing import Optional, Tuple
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired

# ── In-Memory Login Rate Limiting (per IP / Email) ──────────────────────────
# max 5 attempts per 60 seconds
LOGIN_ATTEMPTS = defaultdict(list)
RATE_LIMIT_WINDOW = 60
MAX_LOGIN_ATTEMPTS = 5


def check_rate_limit(key: str) -> bool:
    """Return True if allowed, False if exceeded."""
    now = time.time()
    attempts = [t for t in LOGIN_ATTEMPTS[key] if now - t < RATE_LIMIT_WINDOW]
    LOGIN_ATTEMPTS[key] = attempts
    if len(attempts) >= MAX_LOGIN_ATTEMPTS:
        return False
    LOGIN_ATTEMPTS[key].append(now)
    return True


def clear_rate_limit(key: str):
    LOGIN_ATTEMPTS.pop(key, None)


# ── Password Hashing (bcrypt) ────────────────────────────────────────────────
def hash_password(password: str) -> str:
    pwd_bytes = password.encode("utf-8")
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(pwd_bytes, salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
    except Exception:
        return False


# ── Session & Token Serializer ───────────────────────────────────────────────
def get_serializer(secret: str) -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(secret, salt="videoforge-session")


def create_session_token(user_id: str, secret: str) -> str:
    s = get_serializer(secret)
    return s.dumps({"user_id": user_id, "created_at": time.time()})


def decode_session_token(token: str, secret: str, max_age_days: int = 14) -> Optional[str]:
    s = get_serializer(secret)
    try:
        data = s.loads(token, max_age=max_age_days * 86400)
        return data.get("user_id")
    except (BadSignature, SignatureExpired):
        return None


# ── CSRF Protection ─────────────────────────────────────────────────────────
def generate_csrf_token(secret: str, session_id: str) -> str:
    message = f"csrf:{session_id}".encode("utf-8")
    return hmac.new(secret.encode("utf-8"), message, hashlib.sha256).hexdigest()


def verify_csrf_token(token: str, secret: str, session_id: str) -> bool:
    expected = generate_csrf_token(secret, session_id)
    return hmac.compare_digest(token, expected)
