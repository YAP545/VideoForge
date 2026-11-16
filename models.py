"""Database models and connection using SQLAlchemy.
Configured for SQLite now, with schema design portable to PostgreSQL later.
"""
import os
from datetime import datetime
from pathlib import Path
from sqlalchemy import (
    create_engine,
    Column,
    String,
    Integer,
    Boolean,
    ForeignKey,
    DateTime,
    Text,
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

BASE = Path(__file__).parent
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE / 'videoforge.db'}")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    credits = Column(Integer, default=20, nullable=False)
    is_admin = Column(Boolean, default=False, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    verification_token = Column(String(64), nullable=True)
    reset_token = Column(String(64), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    videos = relationship("Video", back_populates="user", cascade="all, delete-orphan")
    payments = relationship("Payment", cascade="all, delete-orphan")


class Video(Base):
    __tablename__ = "videos"

    id = Column(String(36), primary_key=True)
    user_id = Column(String(36), ForeignKey("users.id"), index=True, nullable=False)
    title = Column(String(255), nullable=False)
    topic = Column(Text, nullable=False)
    length_seconds = Column(Integer, default=15, nullable=False)
    style = Column(String(64), default="2d_cartoon", nullable=False)
    aspect_ratio = Column(String(16), default="16:9", nullable=False)
    caption_lang = Column(String(16), default="none", nullable=False)
    status = Column(String(32), default="planning", nullable=False)  # planning, generating, stitching, done, failed
    video_path = Column(String(255), nullable=True)
    no_music_video_path = Column(String(255), nullable=True)
    srt_path = Column(String(255), nullable=True)
    zip_path = Column(String(255), nullable=True)
    music_filename = Column(String(255), nullable=True)
    keep_consistency = Column(Boolean, default=False, nullable=False)
    progress_text = Column(String(255), nullable=True)
    scenes_json = Column(Text, nullable=True)
    error = Column(Text, nullable=True)
    credits_refunded = Column(Boolean, default=False, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="videos")
    scenes = relationship("Scene", back_populates="video", order_by="Scene.order_idx", cascade="all, delete-orphan")


class Scene(Base):
    __tablename__ = "scenes"

    id = Column(String(36), primary_key=True)
    video_id = Column(String(36), ForeignKey("videos.id"), index=True, nullable=False)
    order_idx = Column(Integer, default=0, nullable=False)
    visual_prompt = Column(Text, nullable=False)
    caption = Column(Text, default="", nullable=False)
    narration = Column(Text, default="", nullable=True)
    is_placeholder = Column(Boolean, default=False, nullable=True)
    is_reused = Column(Boolean, default=False, nullable=True)
    reused_from = Column(Integer, nullable=True)
    status = Column(String(32), default="pending", nullable=False)  # pending, running, done, failed
    clip_path = Column(String(255), nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    video = relationship("Video", back_populates="scenes")


class Payment(Base):
    __tablename__ = "payments"

    id = Column(String(64), primary_key=True)  # Razorpay order id
    user_id = Column(String(36), ForeignKey("users.id"), index=True, nullable=False)
    pack = Column(String(32), nullable=False)
    credits = Column(Integer, nullable=False)
    amount = Column(Integer, nullable=False)  # in paise
    status = Column(String(16), default="created", nullable=False)  # created, paid
    payment_id = Column(String(64), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    Base.metadata.create_all(bind=engine)
    # Safe SQLite column migrations if missing
    from sqlalchemy import text
    with engine.connect() as conn:
        # Check videos columns
        res = conn.execute(text("PRAGMA table_info(videos)"))
        video_cols = {row[1] for row in res.fetchall()}
        if "progress_text" not in video_cols:
            conn.execute(text("ALTER TABLE videos ADD COLUMN progress_text VARCHAR(255)"))
        if "scenes_json" not in video_cols:
            conn.execute(text("ALTER TABLE videos ADD COLUMN scenes_json TEXT"))
        if "credits_refunded" not in video_cols:
            conn.execute(text("ALTER TABLE videos ADD COLUMN credits_refunded BOOLEAN DEFAULT 0"))
        # Check scenes columns
        res = conn.execute(text("PRAGMA table_info(scenes)"))
        scene_cols = {row[1] for row in res.fetchall()}
        if "narration" not in scene_cols:
            conn.execute(text("ALTER TABLE scenes ADD COLUMN narration TEXT"))
        if "is_placeholder" not in scene_cols:
            conn.execute(text("ALTER TABLE scenes ADD COLUMN is_placeholder BOOLEAN DEFAULT 0"))
        if "is_reused" not in scene_cols:
            conn.execute(text("ALTER TABLE scenes ADD COLUMN is_reused BOOLEAN DEFAULT 0"))
        if "reused_from" not in scene_cols:
            conn.execute(text("ALTER TABLE scenes ADD COLUMN reused_from INTEGER"))
        conn.commit()

