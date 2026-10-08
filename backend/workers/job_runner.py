import asyncio
import traceback
from sqlalchemy.orm import Session
from infrastructure.database import SessionLocal
from application.orchestrator import orchestrator


def run_document_job(document_id: str):
    """Executes document processing background task using local database session."""
    db: Session = SessionLocal()
    try:
        print(f"[Worker] Starting background processing for document: {document_id}")
        orchestrator.process_document(db, document_id)
        print(f"[Worker] Successfully completed processing for document: {document_id}")
    except Exception as e:
        print(f"[Worker] Error processing document {document_id}: {e}")
        traceback.print_exc()
    finally:
        db.close()
