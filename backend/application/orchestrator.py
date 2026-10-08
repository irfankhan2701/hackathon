import json
import uuid
import datetime
from typing import Dict, List, Any, Optional
from sqlalchemy.orm import Session

from domain.entities import (
    Document,
    Page,
    DocumentStatus,
    WordEvidence,
    Decision,
    ReviewTask,
)
from infrastructure.database import (
    DocumentModel,
    PageModel,
    RegionModel,
    LineModel,
    WordEvidenceModel,
    DecisionModel,
)
from infrastructure.storage.artifact_store import artifact_store, Artifact
from infrastructure.imaging.preprocessor import process_and_create_artifacts, load_image_bytes
from infrastructure.models.detector import layout_detector, crop_numpy_image
from infrastructure.models.ocr_provider import trocr_provider
from application.fusion import fusion_engine
from application.reranker import context_reranker
from infrastructure.models.verifier import vlm_verifier


class DocumentOrchestrator:
    """
    End-to-end provenance-preserving handwriting document orchestrator.
    Executes full multi-phase pipeline and persists immutable DB & artifact state.
    """

    def process_document(
        self, db: Session, document_id: str
    ) -> Document:
        doc_model = db.query(DocumentModel).filter_by(document_id=document_id).first()
        if not doc_model:
            raise ValueError(f"Document {document_id} not found in database.")

        try:
            doc_model.status = DocumentStatus.PROCESSING.value
            db.commit()

            # Load raw source file bytes
            raw_bytes = artifact_store.get_artifact_bytes(doc_model.source_artifact_id)
            if not raw_bytes:
                raise ValueError("Source artifact binary content not found.")

            # Load images (supports PDF multi-page or single image)
            img_list = load_image_bytes(raw_bytes, doc_model.filename)
            if not img_list:
                raise ValueError("Could not parse image from source file.")

            doc_model.page_count = len(img_list)

            for page_idx, raw_img_np in enumerate(img_list):
                page_num = page_idx + 1

                # Phase 1: Quality Profiling & Preprocessing Artifact Pipeline
                # Create raw image bytes buffer for single page
                from PIL import Image
                import io
                buf = io.BytesIO()
                Image.fromarray(raw_img_np).save(buf, format="PNG")
                p_bytes = buf.getvalue()

                artifacts_map, quality_metrics, deskewed_np, width, height = process_and_create_artifacts(
                    p_bytes, f"page_{page_num}_{doc_model.filename}"
                )

                # Persist Page entity
                page_id = str(uuid.uuid4())
                page_artifacts_json = {k: art.artifact_id for k, art in artifacts_map.items()}

                page_model = PageModel(
                    page_id=page_id,
                    document_id=document_id,
                    page_number=page_num,
                    width=width,
                    height=height,
                    quality_metrics_json=json.dumps(quality_metrics.model_dump()),
                    artifacts_json=json.dumps(page_artifacts_json),
                )
                db.add(page_model)
                db.flush()

                # Phase 2: Layout Intelligence and Document Graph
                regions, lines, line_crops = layout_detector.detect_layout_and_lines(
                    page_id=page_id, img_np=deskewed_np
                )

                # Persist Regions
                for r in regions:
                    r_model = RegionModel(
                        region_id=r.region_id,
                        page_id=page_id,
                        x=r.bounding_box.x,
                        y=r.bounding_box.y,
                        w=r.bounding_box.w,
                        h=r.bounding_box.h,
                        region_type=r.region_type.value,
                        reading_order_index=r.reading_order_index,
                    )
                    db.add(r_model)

                # Persist Lines
                for l in lines:
                    l_model = LineModel(
                        line_id=l.line_id,
                        region_id=l.region_id,
                        page_id=page_id,
                        x=l.bounding_box.x,
                        y=l.bounding_box.y,
                        w=l.bounding_box.w,
                        h=l.bounding_box.h,
                        line_index=l.line_index,
                        reading_order_index=l.reading_order_index,
                        line_crop_artifact_id=l.line_crop_artifact_id,
                        is_crossed_out_candidate=l.is_crossed_out_candidate,
                    )
                    db.add(l_model)
                db.flush()

                # Phase 3 & 4: Multi-variant TrOCR Recognition & Evidence Fusion Engine
                all_page_words: List[WordEvidence] = []
                run_id = str(uuid.uuid4())

                # Load image variant crops for line multi-hypothesis recognition
                clahe_np = load_image_bytes(artifact_store.get_artifact_bytes(artifacts_map["clahe"].artifact_id))[0]
                bin_np = load_image_bytes(artifact_store.get_artifact_bytes(artifacts_map["adaptive_binarized"].artifact_id))[0]
                denoised_np = load_image_bytes(artifact_store.get_artifact_bytes(artifacts_map["denoised"].artifact_id))[0]

                for line in lines:
                    line_hypotheses = []

                    # Crop 4 variants
                    crop_raw = crop_numpy_image(deskewed_np, line.bounding_box)
                    crop_clahe = crop_numpy_image(clahe_np, line.bounding_box)
                    crop_bin = crop_numpy_image(bin_np, line.bounding_box)
                    crop_denoised = crop_numpy_image(denoised_np, line.bounding_box)

                    # Recognize line across 4 variants -> up to 20 hypotheses
                    line_hypotheses.extend(trocr_provider.recognize_crop(crop_raw, line.line_id, run_id, "raw"))
                    line_hypotheses.extend(trocr_provider.recognize_crop(crop_clahe, line.line_id, run_id, "clahe"))
                    line_hypotheses.extend(trocr_provider.recognize_crop(crop_bin, line.line_id, run_id, "adaptive_binarized"))
                    line_hypotheses.extend(trocr_provider.recognize_crop(crop_denoised, line.line_id, run_id, "denoised"))

                    # Fuse hypotheses into WordEvidence records
                    word_evs = fusion_engine.fuse_line_hypotheses(
                        line=line,
                        hypotheses=line_hypotheses,
                        quality_metrics=quality_metrics,
                    )
                    all_page_words.extend(word_evs)

                # Phase 5: Safe Context Reranking and Format Normalization
                reranked_words, audit_events = context_reranker.rerank_and_format_words(all_page_words)

                # Persist Word Evidence & Decisions
                for word in reranked_words:
                    cands_json = json.dumps([c.model_dump() for c in word.candidates])
                    formula_json = json.dumps(word.formula_components.model_dump()) if word.formula_components else "{}"

                    word_model = WordEvidenceModel(
                        word_id=word.word_id,
                        line_id=word.line_id,
                        word_index=word.word_index,
                        text=word.text,
                        x=word.bounding_box.x,
                        y=word.bounding_box.y,
                        w=word.bounding_box.w,
                        h=word.bounding_box.h,
                        source_crop_id=word.source_crop_id,
                        candidates_json=cands_json,
                        confidence=word.confidence,
                        uncertainty=word.uncertainty,
                        decision_status=word.decision_status.value,
                        decision_reason=word.decision_reason,
                        raw_text=word.raw_text,
                        corrected_text=word.corrected_text,
                        is_edited=word.is_edited,
                        formula_components_json=formula_json,
                    )
                    db.add(word_model)

                    # Create Decision log
                    dec_model = DecisionModel(
                        decision_id=str(uuid.uuid4()),
                        word_id=word.word_id,
                        decision_status=word.decision_status.value,
                        final_score=word.confidence,
                        uncertainty_score=word.uncertainty,
                        formula_components_json=formula_json,
                        timestamp=datetime.datetime.utcnow().isoformat(),
                    )
                    db.add(dec_model)

            doc_model.status = DocumentStatus.COMPLETED.value
            db.commit()

            return Document(
                document_id=doc_model.document_id,
                filename=doc_model.filename,
                file_type=doc_model.file_type,
                created_at=doc_model.created_at,
                status=DocumentStatus.COMPLETED,
                source_artifact_id=doc_model.source_artifact_id,
                page_count=doc_model.page_count,
            )

        except Exception as e:
            db.rollback()
            doc_model.status = DocumentStatus.FAILED.value
            doc_model.error_message = str(e)
            db.commit()
            print(f"Document processing failed for {document_id}: {e}")
            raise e


orchestrator = DocumentOrchestrator()
