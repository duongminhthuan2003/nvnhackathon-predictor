# app/api.py
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.services.predict import (
    load_model,
    load_label_map,
    predict_from_bytes,
)

app = FastAPI(title="Vietnamese Sign Language API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://naver-hackathon-project.vercel.app",
        "https://naver-hackathon-project.vercel.app/record",
        "https://www.naver-hackathon-project.vercel.app",
        "https://www.naver-hackathon-project.vercel.app/record",
	"https://naver-hackathon-project.vercel.app",
	"https://gitinitpredictor.duckdns.org",
	"https://www.gitinitpredictor.duckdns.org",
	"https://gitinitpredictor.duckdns.org/predict",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174",
	"*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khởi động: nạp model và label map vào bộ nhớ
load_model()
load_label_map()

ALLOWED_VIDEO_TYPES = {
    "video/mp4": ".mp4",
    "video/x-msvideo": ".avi",
    "video/webm": ".webm",
}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    content_type = file.content_type or ""
    if content_type not in ALLOWED_VIDEO_TYPES:
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ MP4/AVI/WEBM")

    video_bytes = await file.read()
    if not video_bytes:
        raise HTTPException(status_code=400, detail="File rỗng")

    suffix = ""
    if file.filename:
        suffix = Path(file.filename).suffix.lower()

    if suffix not in ALLOWED_VIDEO_TYPES.values():
        suffix = ALLOWED_VIDEO_TYPES[content_type]

    try:
        result = predict_from_bytes(video_bytes, suffix=suffix)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Lỗi nội bộ: {exc}")

    return result