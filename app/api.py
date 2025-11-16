# app/api.py
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
        "https://www.naver-hackathon-project.vercel.app",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174",
    ],
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

# Khởi động: nạp model và label map vào bộ nhớ
load_model()
load_label_map()

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ("video/mp4", "video/x-msvideo"):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ MP4/AVI")

    video_bytes = await file.read()
    if not video_bytes:
        raise HTTPException(status_code=400, detail="File rỗng")

    try:
        result = predict_from_bytes(video_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Lỗi nội bộ: {exc}")

    return result