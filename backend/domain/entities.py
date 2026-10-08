import datetime
import uuid
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class DocumentStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class QualityClass(str, Enum):
    EXCELLENT = "EXCELLENT"
    GOOD = "GOOD"
    FAIR = "FAIR"
    POOR = "POOR"


class RegionType(str, Enum):
    MAIN_HANDWRITING = "main_handwriting"
    PRINTED_TEXT = "printed_text"
    MARGIN_NOTE = "margin_note"
    TABLE = "table"
    CROSSED_OUT_CANDIDATE = "crossed_out_candidate"
    SIGNATURE = "signature"
    UNKNOWN = "unknown"


class DecisionStatus(str, Enum):
    ACCEPTED = "ACCEPTED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    ILLEGIBLE = "ILLEGIBLE"
    CROSSED_OUT = "CROSSED_OUT"


class BoundingBox(BaseModel):
    x: float
    y: float
    w: float
    h: float


class Artifact(BaseModel):
    artifact_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sha256_hash: str
    file_path: str
    file_type: str
    source_artifact_id: Optional[str] = None
    transformation_name: str
    settings: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.datetime.utcnow().isoformat())


class QualityMetrics(BaseModel):
    blur_score: float
    contrast_score: float
    skew_angle: float
    resolution: List[int]  # [width, height]
    noise_score: float
    quality_class: QualityClass


class Region(BaseModel):
    region_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    page_id: str
    bounding_box: BoundingBox
    region_type: RegionType
    reading_order_index: int = 0


class Line(BaseModel):
    line_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    region_id: str
    page_id: str
    bounding_box: BoundingBox
    line_index: int
    reading_order_index: int
    line_crop_artifact_id: Optional[str] = None
    is_crossed_out_candidate: bool = False


class RecognitionRun(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    line_id: str
    model_name: str
    model_version: str
    config: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.datetime.utcnow().isoformat())


class Hypothesis(BaseModel):
    hypothesis_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    line_id: str
    run_id: str
    text: str
    score: float
    variant_name: str  # raw, clahe, adaptive_binarized, denoised
    token_confidences: List[float] = Field(default_factory=list)
    model_name: str = "microsoft/trocr-large-handwritten"


class WordCandidate(BaseModel):
    text: str
    score: float
    variant_name: str
    model_name: str


class DecisionFormulaComponents(BaseModel):
    visual_score: float
    variant_agreement: float
    character_stability: float
    context_score: float
    verifier_agreement: float
    decoder_entropy: float
    ensemble_disagreement: float
    image_quality_risk: float
    verifier_disagreement: float


class WordEvidence(BaseModel):
    word_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    line_id: str
    word_index: int
    text: str
    bounding_box: BoundingBox
    source_crop_id: Optional[str] = None
    candidates: List[WordCandidate] = Field(default_factory=list)
    confidence: float
    uncertainty: float
    decision_status: DecisionStatus
    decision_reason: str
    raw_text: str
    corrected_text: Optional[str] = None
    is_edited: bool = False
    formula_components: Optional[DecisionFormulaComponents] = None


class Decision(BaseModel):
    decision_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    word_id: str
    decision_status: DecisionStatus
    final_score: float
    uncertainty_score: float
    formula_components: DecisionFormulaComponents
    timestamp: str = Field(default_factory=lambda: datetime.datetime.utcnow().isoformat())


class Page(BaseModel):
    page_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str
    page_number: int
    width: int
    height: int
    quality_metrics: Optional[QualityMetrics] = None
    artifacts: Dict[str, str] = Field(default_factory=dict)  # variant_name -> artifact_id
    regions: List[Region] = Field(default_factory=list)
    lines: List[Line] = Field(default_factory=list)


class Document(BaseModel):
    document_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    file_type: str
    created_at: str = Field(default_factory=lambda: datetime.datetime.utcnow().isoformat())
    status: DocumentStatus = DocumentStatus.PENDING
    error_message: Optional[str] = None
    source_artifact_id: str
    page_count: int = 1
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ReviewTask(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str
    word_id: str
    status: str = "PENDING"
    assigned_user: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.datetime.utcnow().isoformat())
