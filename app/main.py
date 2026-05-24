from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

from app.review_engine import ReviewRequest, ReviewResponse, generate_reviews, model_info as engine_model_info


app = FastAPI(
    title="NaijaRec Review Generator",
    description="Persona-grounded review and rating generator for Nigerian-diaspora food recommendation demos.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/model-info")
def model_info():
    return engine_model_info()


@app.post("/api/generate-review", response_model=ReviewResponse)
def generate_review(request: ReviewRequest):
    return generate_reviews(request)
