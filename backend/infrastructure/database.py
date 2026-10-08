import json
import os
from typing import Dict, List, Optional, Any
from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    Boolean,
    Text,
    DateTime,
    ForeignKey,
    create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
import datetime

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/scribeproof.db")

os.makedirs("./data", exist_ok=True)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class DocumentModel(Base):
    __tablename__ = "documents"

    document_id = Column(String, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    created_at = Column(String, nullable=False)
    status = Column(String, nullable=False, default="PENDING")
    error_message = Column(Text, nullable=True)
    source_artifact_id = Column(String, nullable=False)
    page_count = Column(Integer, default=1)
    metadata_json = Column(Text, default="{}")

    pages = relationship("PageModel", back_populates="document", cascade="all, delete-orphan")


class PageModel(Base):
    __tablename__ = "pages"

    page_id = Column(String, primary_key=True, index=True)
    document_id = Column(String, ForeignKey("documents.document_id"), nullable=False)
    page_number = Column(Integer, nullable=False)
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    quality_metrics_json = Column(Text, nullable=True)
    artifacts_json = Column(Text, default="{}")

    document = relationship("DocumentModel", back_populates="pages")
    regions = relationship("RegionModel", back_populates="page", cascade="all, delete-orphan")
    lines = relationship("LineModel", back_populates="page", cascade="all, delete-orphan")


class RegionModel(Base):
    __tablename__ = "regions"

    region_id = Column(String, primary_key=True, index=True)
    page_id = Column(String, ForeignKey("pages.page_id"), nullable=False)
    x = Column(Float, nullable=False)
    y = Column(Float, nullable=False)
    w = Column(Float, nullable=False)
    h = Column(Float, nullable=False)
    region_type = Column(String, nullable=False)
    reading_order_index = Column(Integer, default=0)

    page = relationship("PageModel", back_populates="regions")
    lines = relationship("LineModel", back_populates="region", cascade="all, delete-orphan")


class LineModel(Base):
    __tablename__ = "lines"

    line_id = Column(String, primary_key=True, index=True)
    region_id = Column(String, ForeignKey("regions.region_id"), nullable=False)
    page_id = Column(String, ForeignKey("pages.page_id"), nullable=False)
    x = Column(Float, nullable=False)
    y = Column(Float, nullable=False)
    w = Column(Float, nullable=False)
    h = Column(Float, nullable=False)
    line_index = Column(Integer, nullable=False)
    reading_order_index = Column(Integer, nullable=False)
    line_crop_artifact_id = Column(String, nullable=True)
    is_crossed_out_candidate = Column(Boolean, default=False)

    page = relationship("PageModel", back_populates="lines")
    region = relationship("RegionModel", back_populates="lines")
    words = relationship("WordEvidenceModel", back_populates="line", cascade="all, delete-orphan")


class WordEvidenceModel(Base):
    __tablename__ = "word_evidences"

    word_id = Column(String, primary_key=True, index=True)
    line_id = Column(String, ForeignKey("lines.line_id"), nullable=False)
    word_index = Column(Integer, nullable=False)
    text = Column(String, nullable=False)
    x = Column(Float, nullable=False)
    y = Column(Float, nullable=False)
    w = Column(Float, nullable=False)
    h = Column(Float, nullable=False)
    source_crop_id = Column(String, nullable=True)
    candidates_json = Column(Text, default="[]")
    confidence = Column(Float, nullable=False)
    uncertainty = Column(Float, nullable=False)
    decision_status = Column(String, nullable=False)
    decision_reason = Column(Text, nullable=False)
    raw_text = Column(String, nullable=False)
    corrected_text = Column(String, nullable=True)
    is_edited = Column(Boolean, default=False)
    formula_components_json = Column(Text, nullable=True)

    line = relationship("LineModel", back_populates="words")


class DecisionModel(Base):
    __tablename__ = "decisions"

    decision_id = Column(String, primary_key=True, index=True)
    word_id = Column(String, ForeignKey("word_evidences.word_id"), nullable=False)
    decision_status = Column(String, nullable=False)
    final_score = Column(Float, nullable=False)
    uncertainty_score = Column(Float, nullable=False)
    formula_components_json = Column(Text, nullable=False)
    timestamp = Column(String, nullable=False)


class ReviewTaskModel(Base):
    __tablename__ = "review_tasks"

    task_id = Column(String, primary_key=True, index=True)
    document_id = Column(String, ForeignKey("documents.document_id"), nullable=False)
    word_id = Column(String, ForeignKey("word_evidences.word_id"), nullable=False)
    status = Column(String, default="PENDING")
    assigned_user = Column(String, nullable=True)
    created_at = Column(String, nullable=False)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
