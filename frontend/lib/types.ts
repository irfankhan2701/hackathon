export type DocumentStatus = "PENDING" | "PROCESSING" | "COMPLETED" | "FAILED";
export type DecisionStatus = "ACCEPTED" | "REVIEW_REQUIRED" | "ILLEGIBLE" | "CROSSED_OUT";
export type QualityClass = "EXCELLENT" | "GOOD" | "FAIR" | "POOR";
export type RegionType =
  | "main_handwriting"
  | "printed_text"
  | "margin_note"
  | "table"
  | "crossed_out_candidate"
  | "signature"
  | "unknown";

export interface BoundingBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface WordCandidate {
  text: string;
  score: number;
  variant_name: string;
  model_name: string;
}

export interface DecisionFormulaComponents {
  visual_score: number;
  variant_agreement: number;
  character_stability: number;
  context_score: number;
  verifier_agreement: number;
  decoder_entropy: number;
  ensemble_disagreement: number;
  image_quality_risk: number;
  verifier_disagreement: number;
}

export interface WordEvidence {
  word_id: string;
  line_id: string;
  word_index: number;
  text: string;
  bounding_box: BoundingBox;
  source_crop_id?: string;
  candidates: WordCandidate[];
  confidence: number;
  uncertainty: number;
  decision_status: DecisionStatus;
  decision_reason: string;
  raw_text: string;
  corrected_text?: string;
  is_edited: boolean;
  formula_components?: DecisionFormulaComponents;
}

export interface Line {
  line_id: string;
  region_id: string;
  page_id: string;
  bounding_box: BoundingBox;
  line_index: number;
  reading_order_index: number;
  line_crop_artifact_id?: string;
  is_crossed_out_candidate: boolean;
  words: WordEvidence[];
}

export interface Region {
  region_id: string;
  page_id: string;
  bounding_box: BoundingBox;
  region_type: RegionType;
  reading_order_index: number;
  lines: Line[];
}

export interface QualityMetrics {
  blur_score: number;
  contrast_score: number;
  skew_angle: number;
  resolution: [number, number];
  noise_score: number;
  quality_class: QualityClass;
}

export interface PageResult {
  page_id: string;
  page_number: number;
  width: number;
  height: number;
  quality_metrics?: QualityMetrics;
  artifacts: Record<string, string>; // variant_name -> artifact_id
  regions: Region[];
}

export interface DocumentSummaryStats {
  total_words: number;
  accepted_words: number;
  review_required_words: number;
  illegible_words: number;
  crossed_out_words: number;
  coverage_percentage: number;
}

export interface DocumentResult {
  document_id: string;
  filename: string;
  file_type: string;
  created_at: string;
  status: DocumentStatus;
  error_message?: string;
  source_artifact_id: string;
  page_count: number;
  pages: PageResult[];
  summary_stats: DocumentSummaryStats;
}

export interface DocumentListItem {
  document_id: string;
  filename: string;
  file_type: string;
  created_at: string;
  status: DocumentStatus;
  page_count: number;
  error_message?: string;
}

export interface EvaluationReport {
  status: string;
  sample_evaluation: {
    overall_cer: number;
    overall_wer: number;
    accepted_cer: number;
    accepted_wer: number;
    coverage_percentage: number;
    selective_risk: number;
    illegible_precision: number;
    illegible_recall: number;
    crossed_out_precision: number;
    crossed_out_recall: number;
    processing_time_sec: number;
  };
  ablation_report: Array<{
    mode: string;
    cer: number;
    wer: number;
    accepted_wer: number;
    coverage_pct: number;
    selective_risk: number;
    confident_errors: number;
    description: string;
  }>;
}
