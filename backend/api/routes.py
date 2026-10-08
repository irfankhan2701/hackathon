import json
import uuid
import datetime
import os
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks, Response
from fastapi.responses import FileResponse, Response
from sqlalchemy.orm import Session

from infrastructure.database import (
    get_db,
    DocumentModel,
    PageModel,
    RegionModel,
    LineModel,
    WordEvidenceModel,
    DecisionModel,
)
from infrastructure.storage.artifact_store import artifact_store
from api.schemas import (
    DocumentUploadResponse,
    DocumentStatusResponse,
    DocumentResultResponse,
    PageResultSchema,
    RegionSchema,
    LineSchema,
    WordEvidenceSchema,
    QualityMetricsSchema,
    WordUpdateRequest,
)
from workers.job_runner import run_document_job

router = APIRouter()


@router.get("/health")
def get_health():
    return {
        "status": "healthy",
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "service": "ScribeProof Handwriting Intelligence API",
    }


@router.post("/documents", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename missing.")

    ext = file.filename.split(".")[-1].lower()
    if ext not in ["png", "jpg", "jpeg", "pdf", "tif", "tiff"]:
        raise HTTPException(status_code=400, detail="Unsupported file format. Supported: PNG, JPG, JPEG, PDF.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Save source artifact content-addressed SHA-256
    source_artifact = artifact_store.save_bytes(
        content=content,
        file_type=ext,
        transformation_name="raw",
        settings={"filename": file.filename},
    )

    doc_id = str(uuid.uuid4())
    doc_model = DocumentModel(
        document_id=doc_id,
        filename=file.filename,
        file_type=ext,
        created_at=datetime.datetime.utcnow().isoformat(),
        status="PENDING",
        source_artifact_id=source_artifact.artifact_id,
        page_count=1,
    )
    db.add(doc_model)
    db.commit()

    # Trigger async background job
    background_tasks.add_task(run_document_job, doc_id)

    return DocumentUploadResponse(
        document_id=doc_id,
        filename=file.filename,
        file_type=ext,
        status="PENDING",
        source_artifact_id=source_artifact.artifact_id,
        created_at=doc_model.created_at,
    )


@router.get("/documents")
def list_documents(db: Session = Depends(get_db)):
    docs = db.query(DocumentModel).order_by(DocumentModel.created_at.desc()).all()
    return [
        {
            "document_id": d.document_id,
            "filename": d.filename,
            "file_type": d.file_type,
            "created_at": d.created_at,
            "status": d.status,
            "page_count": d.page_count,
            "error_message": d.error_message,
        }
        for d in docs
    ]


@router.get("/documents/{document_id}")
def get_document(document_id: str, db: Session = Depends(get_db)):
    doc = db.query(DocumentModel).filter_by(document_id=document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found.")
    return {
        "document_id": doc.document_id,
        "filename": doc.filename,
        "file_type": doc.file_type,
        "created_at": doc.created_at,
        "status": doc.status,
        "page_count": doc.page_count,
        "source_artifact_id": doc.source_artifact_id,
        "error_message": doc.error_message,
    }


@router.get("/documents/{document_id}/status", response_model=DocumentStatusResponse)
def get_document_status(document_id: str, db: Session = Depends(get_db)):
    doc = db.query(DocumentModel).filter_by(document_id=document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found.")
    return DocumentStatusResponse(
        document_id=doc.document_id,
        status=doc.status,
        error_message=doc.error_message,
        page_count=doc.page_count,
    )


@router.get("/documents/{document_id}/result", response_model=DocumentResultResponse)
def get_document_result(document_id: str, db: Session = Depends(get_db)):
    doc = db.query(DocumentModel).filter_by(document_id=document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found.")

    pages_models = db.query(PageModel).filter_by(document_id=document_id).all()
    pages_schemas: List[PageResultSchema] = []

    total_words = 0
    accepted_words = 0
    review_words = 0
    illegible_words = 0
    crossed_out_words = 0

    for p in pages_models:
        qm_dict = json.loads(p.quality_metrics_json) if p.quality_metrics_json else {}
        artifacts_dict = json.loads(p.artifacts_json) if p.artifacts_json else {}

        regions_models = db.query(RegionModel).filter_by(page_id=p.page_id).order_by(RegionModel.reading_order_index).all()
        regions_schemas: List[RegionSchema] = []

        for r in regions_models:
            lines_models = db.query(LineModel).filter_by(region_id=r.region_id).order_by(LineModel.reading_order_index).all()
            lines_schemas: List[LineSchema] = []

            for l in lines_models:
                words_models = db.query(WordEvidenceModel).filter_by(line_id=l.line_id).order_by(WordEvidenceModel.word_index).all()
                words_schemas: List[WordEvidenceSchema] = []

                for w in words_models:
                    total_words += 1
                    status_val = w.decision_status
                    if status_val == "ACCEPTED":
                        accepted_words += 1
                    elif status_val == "REVIEW_REQUIRED":
                        review_words += 1
                    elif status_val == "ILLEGIBLE":
                        illegible_words += 1
                    elif status_val == "CROSSED_OUT":
                        crossed_out_words += 1

                    cands_list = json.loads(w.candidates_json) if w.candidates_json else []
                    formula_dict = json.loads(w.formula_components_json) if w.formula_components_json else None

                    words_schemas.append(
                        WordEvidenceSchema(
                            word_id=w.word_id,
                            line_id=w.line_id,
                            word_index=w.word_index,
                            text=w.corrected_text if w.is_edited and w.corrected_text else w.text,
                            bounding_box={"x": w.x, "y": w.y, "w": w.w, "h": w.h},
                            source_crop_id=w.source_crop_id,
                            candidates=cands_list,
                            confidence=w.confidence,
                            uncertainty=w.uncertainty,
                            decision_status=w.decision_status,
                            decision_reason=w.decision_reason,
                            raw_text=w.raw_text,
                            corrected_text=w.corrected_text,
                            is_edited=w.is_edited,
                            formula_components=formula_dict,
                        )
                    )

                lines_schemas.append(
                    LineSchema(
                        line_id=l.line_id,
                        region_id=l.region_id,
                        page_id=l.page_id,
                        bounding_box={"x": l.x, "y": l.y, "w": l.w, "h": l.h},
                        line_index=l.line_index,
                        reading_order_index=l.reading_order_index,
                        line_crop_artifact_id=l.line_crop_artifact_id,
                        is_crossed_out_candidate=l.is_crossed_out_candidate,
                        words=words_schemas,
                    )
                )

            regions_schemas.append(
                RegionSchema(
                    region_id=r.region_id,
                    page_id=r.page_id,
                    bounding_box={"x": r.x, "y": r.y, "w": r.w, "h": r.h},
                    region_type=r.region_type,
                    reading_order_index=r.reading_order_index,
                    lines=lines_schemas,
                )
            )

        pages_schemas.append(
            PageResultSchema(
                page_id=p.page_id,
                page_number=p.page_number,
                width=p.width,
                height=p.height,
                quality_metrics=QualityMetricsSchema(**qm_dict) if qm_dict else None,
                artifacts=artifacts_dict,
                regions=regions_schemas,
            )
        )

    coverage_pct = round((accepted_words / max(1, total_words)) * 100.0, 1)

    return DocumentResultResponse(
        document_id=doc.document_id,
        filename=doc.filename,
        file_type=doc.file_type,
        created_at=doc.created_at,
        status=doc.status,
        error_message=doc.error_message,
        source_artifact_id=doc.source_artifact_id,
        page_count=doc.page_count,
        pages=pages_schemas,
        summary_stats={
            "total_words": total_words,
            "accepted_words": accepted_words,
            "review_required_words": review_words,
            "illegible_words": illegible_words,
            "crossed_out_words": crossed_out_words,
            "coverage_percentage": coverage_pct,
        },
    )


@router.patch("/words/{word_id}")
def update_word(word_id: str, req: WordUpdateRequest, db: Session = Depends(get_db)):
    w_model = db.query(WordEvidenceModel).filter_by(word_id=word_id).first()
    if not w_model:
        raise HTTPException(status_code=404, detail=f"Word {word_id} not found.")

    w_model.corrected_text = req.corrected_text
    w_model.is_edited = True
    if req.decision_status:
        w_model.decision_status = req.decision_status

    db.commit()

    return {
        "word_id": w_model.word_id,
        "text": w_model.corrected_text,
        "decision_status": w_model.decision_status,
        "is_edited": True,
    }


@router.get("/artifacts/{artifact_id}")
def get_artifact_file(artifact_id: str):
    file_path = artifact_store.get_artifact_path(artifact_id)
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Artifact {artifact_id} not found.")

    art = artifact_store.get_artifact(artifact_id)
    ext = art.file_type.lower() if art else "png"
    media_type = "image/png"
    if ext in ["jpg", "jpeg"]:
        media_type = "image/jpeg"
    elif ext == "pdf":
        media_type = "application/pdf"

    return FileResponse(file_path, media_type=media_type)


@router.get("/documents/{document_id}/export")
def export_document(
    document_id: str, format: str = "md", db: Session = Depends(get_db)
):
    doc_res = get_document_result(document_id, db)
    if format not in ["md", "json", "txt"]:
        raise HTTPException(status_code=400, detail="Invalid format. Supported: md, json, txt")

    if format == "json":
        return doc_res.model_dump()

    lines_text = []
    for page in doc_res.pages:
        lines_text.append(f"--- Page {page.page_number} ---")
        for region in page.regions:
            if region.region_type == "margin_note":
                lines_text.append("\n[Margin Note]")
            for line in region.lines:
                line_words = []
                for w in line.words:
                    word_display = w.text
                    if w.decision_status == "CROSSED_OUT":
                        word_display = f"~~{w.text}~~"
                    elif w.decision_status == "ILLEGIBLE":
                        word_display = "[ILLEGIBLE]"
                    elif w.decision_status == "REVIEW_REQUIRED":
                        word_display = f"[{w.text}?]"
                    line_words.append(word_display)
                lines_text.append(" ".join(line_words))

    content = "\n".join(lines_text)

    if format == "txt":
        return Response(content=content, media_type="text/plain")

    # Markdown format
    md_header = f"# ScribeProof Digitized Document\n**Document ID**: `{doc_res.document_id}`\n**Source File**: `{doc_res.filename}`\n**Coverage**: `{doc_res.summary_stats.get('coverage_percentage')}%`\n\n---\n\n"
    return Response(content=md_header + content, media_type="text/markdown")
