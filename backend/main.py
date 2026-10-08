import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from infrastructure.database import init_db
from api.routes import router as api_router
from evaluation.evaluator import evaluator
from evaluation.ablation import ablation_runner

app = FastAPI(
    title="ScribeProof - Provenance-Preserving Handwriting Intelligence Stack",
    description="HNX26EPS04 Extreme Bad-Handwriting Digitizing Stack API backend",
    version="1.0.0",
)

# Enable CORS for Next.js app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")


@app.on_event("startup")
def on_startup():
    init_db()
    print("Database tables initialized successfully.")


@app.post("/api/evaluate")
def run_evaluation():
    """Runs CER, WER, and ablation study report."""
    ablation_report = ablation_runner.run_ablation_study()
    sample_eval = evaluator.evaluate_transcription(
        predicted_words=[],
        ground_truth="Patient: Sarah Connor Rx: Amoxicillin 500 mg Take 2 tablets daily for 7 days Date: 12/03/2026 BP: 120/80 mmHg",
        processing_time_sec=0.842,
    )
    return {
        "status": "success",
        "sample_evaluation": sample_eval,
        "ablation_report": ablation_report,
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
