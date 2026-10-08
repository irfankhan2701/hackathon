import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class WordUpdateRequest(BaseModel):
    corrected_text: str
    decision_status: Optional[str] = None


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    file_type: str
    status: str
    source_artifact_id: str
    created_at: str


class DocumentStatusResponse(BaseModel):
    document_id: str
    status: str
    error_message: Optional[str] = None
    page_count: int


class BoundingBoxSchema(BaseModel):
    x: float
    y: float
    w: float
    h: float


class WordCandidateSchema(BaseModel):
    text: str
    score: float
    variant_name: str
    model_name: str


class DecisionFormulaComponentsSchema(BaseModel):
    visual_score: float
    variant_agreement: float
    character_stability: float
    context_score: float
    verifier_agreement: float
    decoder_entropy: float
    ensemble_disagreement: float
    image_quality_risk: float
    verifier_disagreement: float


class WordEvidenceSchema(BaseModel):
    word_id: str
    line_id: str
    word_index: int
    text: str
    bounding_box: BoundingBoxSchema
    source_crop_id: Optional[str] = None
    candidates: List[WordCandidateSchema] = []
    confidence: float
    uncertainty: float
    decision_status: str
    decision_reason: str
    raw_text: str
    corrected_text: Optional[str] = None
    is_edited: bool = False
    formula_components: Optional[DecisionFormulaComponentsSchema] = None


class LineSchema(BaseModel):
    line_id: str
    region_id: str
    page_id: str
    bounding_box: BoundingBoxSchema
    line_index: int
    reading_order_index: int
    line_crop_artifact_id: Optional[str] = None
    is_crossed_out_candidate: bool = False
    words: List[WordEvidenceSchema] = []


class RegionSchema(BaseModel):
    region_id: str
    page_id: str
    bounding_box: BoundingBoxSchema
    region_type: str
    reading_order_index: int
    lines: List[LineSchema] = []


class QualityMetricsSchema(BaseModel):
    blur_score: float
    contrast_score: float
    skew_angle: float
    resolution: List[int]
    noise_score: float
    quality_class: str


class PageResultSchema(BaseModel):
    page_id: str
    page_number: int
    width: int
    height: int
    quality_metrics: Optional[QualityMetricsSchema] = None
    artifacts: Dict[str, str] = {}
    regions: List[RegionSchema] = []


class DocumentResultResponse(BaseModel):
    document_id: str
    filename: str
    file_type: str
    created_at: str
    status: str
    error_message: Optional[str] = None
    source_artifact_id: str
    page_count: int
    pages: List[PageResultSchema] = []
    summary_stats: Dict[str, Any] = {}
