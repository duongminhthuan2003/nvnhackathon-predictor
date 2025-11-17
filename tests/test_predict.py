import pytest
from fastapi.testclient import TestClient

from app.api import app
from app.services import predictor

client = TestClient(app)


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_predict_success(monkeypatch):
    def fake_predict_from_bytes(data: bytes, suffix: str = ".mp4"):
        assert isinstance(data, bytes)
        assert suffix == ".mp4"
        return {
            "label": "xin_chao",
            "confidence": 0.95,
            "probs": [0.95, 0.05],
        }

    monkeypatch.setattr(predictor, "predict_from_bytes", fake_predict_from_bytes)

    files = {"file": ("sample.mp4", b"fake-binary", "video/mp4")}
    resp = client.post("/predict", files=files)

    assert resp.status_code == 200
    json_data = resp.json()
    assert json_data["label"] == "xin_chao"
    assert pytest.approx(json_data["confidence"], rel=1e-6) == 0.95


def test_predict_rejects_invalid_content_type(monkeypatch):
    monkeypatch.setattr(
        predictor,
        "predict_from_bytes",
        lambda data, suffix=".mp4": {"label": "noop", "confidence": 1.0, "probs": []},
    )

    files = {"file": ("sample.txt", b"text", "text/plain")}
    resp = client.post("/predict", files=files)

    assert resp.status_code == 400
    assert resp.json()["detail"] == "Chỉ hỗ trợ MP4/AVI/WEBM"


def test_predict_handles_empty_file(monkeypatch):
    monkeypatch.setattr(
        predictor,
        "predict_from_bytes",
        lambda data, suffix=".mp4": {"label": "noop", "confidence": 1.0, "probs": []},
    )

    files = {"file": ("empty.mp4", b"", "video/mp4")}
    resp = client.post("/predict", files=files)

    assert resp.status_code == 400
    assert resp.json()["detail"] == "File rỗng"


def test_predict_accepts_webm(monkeypatch):
    def fake_predict_from_bytes(data: bytes, suffix: str = ".mp4"):
        assert suffix == ".webm"
        return {
            "label": "xin_chao",
            "confidence": 0.9,
            "probs": [0.9, 0.1],
        }

    monkeypatch.setattr(predictor, "predict_from_bytes", fake_predict_from_bytes)

    files = {"file": ("sample.webm", b"fake-binary", "video/webm")}
    resp = client.post("/predict", files=files)

    assert resp.status_code == 200
    json_data = resp.json()
    assert json_data["label"] == "xin_chao"