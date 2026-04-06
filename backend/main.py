import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from download_model import download_weights
from model import load_model, cleanup, predict_from_bytes

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_SIZE = 10 * 1024 * 1024  # 10 MB


class PredictionResponse(BaseModel):
    score: float            # ensemble fake probability (0-1)
    pixel_score: float      # EfficientNet pixel-domain score
    freq_score: float       # FFT frequency-domain score
    confidence: float
    heatmap_base64: str     # Grad-CAM overlay as base64 PNG
    face_detected: bool
    message: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    download_weights()
    load_model()
    yield
    cleanup()


app = FastAPI(title="Deepfake Detector API", lifespan=lifespan)

# In production on HF Spaces the frontend is served from a *.hf.space domain.
# During local dev it runs on localhost:5173.
CORS_ORIGINS = os.environ.get(
    "CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_origin_regex=r"https://.*\.hf\.space",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    from model import _model_state
    loaded = _model_state is not None
    device = str(_model_state["device"]) if loaded else "unknown"
    return {"status": "healthy", "model_loaded": loaded, "device": device}


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Accepted: JPEG, PNG, WebP.",
        )

    image_bytes = await file.read()
    if len(image_bytes) > MAX_SIZE:
        raise HTTPException(status_code=400, detail="File too large. Max size is 10 MB.")

    try:
        result = predict_from_bytes(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    face_msg = (
        "A face was detected and used for analysis."
        if result["face_detected"]
        else "No face detected; the full image was analyzed."
    )

    return PredictionResponse(
        score=result["score"],
        pixel_score=result["pixel_score"],
        freq_score=result["freq_score"],
        confidence=result["confidence"],
        heatmap_base64=result["heatmap_base64"],
        face_detected=result["face_detected"],
        message=f"Image analyzed successfully. {face_msg}",
    )
