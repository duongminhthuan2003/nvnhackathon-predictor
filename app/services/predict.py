# app/services/predictor.py
import json
import os
os.environ["MEDIAPIPE_DISABLE_SOUNDDEVICE"] = "1"

import tempfile
from typing import Dict, List, Tuple

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from scipy.interpolate import interp1d

# ==== CẤU HÌNH THREAD CHO TF & OPENCV ====
_tf_inter_threads = int(os.getenv("TF_INTER_THREADS", "4"))
_tf_intra_threads = int(os.getenv("TF_INTRA_THREADS", "4"))
tf.config.threading.set_inter_op_parallelism_threads(_tf_inter_threads)
tf.config.threading.set_intra_op_parallelism_threads(_tf_intra_threads)
cv2.setNumThreads(int(os.getenv("OPENCV_THREADS", "4")))

# ==== HẰNG SỐ LANDMARK ====
mp_holistic = mp.solutions.holistic
N_UPPER_BODY_POSE_LANDMARKS = 25
N_HAND_LANDMARKS = 21
N_TOTAL_LANDMARKS = N_UPPER_BODY_POSE_LANDMARKS + 2 * N_HAND_LANDMARKS

# ==== BIẾN CACHE ====
_model: tf.keras.Model | None = None
_label_map: Dict[str, int] | None = None
_inv_label_map: Dict[int, str] | None = None


def load_model(model_path: str = "Models/checkpoints/final_model.keras") -> tf.keras.Model:
    """
    Tải model Keras và cache lại cho toàn bộ process.
    """
    global _model
    if _model is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không tìm thấy model tại {model_path}")
        _model = tf.keras.models.load_model(model_path)
    return _model


def load_label_map(label_path: str = "Logs/label_map.json") -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Tải label_map và tạo bản đảo ngược.
    """
    global _label_map, _inv_label_map
    if _label_map is None or _inv_label_map is None:
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Không tìm thấy label_map tại {label_path}")
        with open(label_path, "r", encoding="utf-8") as f:
            _label_map = json.load(f)
        _inv_label_map = {v: k for k, v in _label_map.items()}
    return _label_map, _inv_label_map


def create_holistic() -> mp_holistic.Holistic:
    """
    Khởi tạo Mediapipe Holistic theo cấu hình mặc định đã tối ưu cho inference.
    """
    return mp_holistic.Holistic(
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
        model_complexity=0,
        smooth_landmarks=True,
        static_image_mode=False,
    )


def mediapipe_detection(image: np.ndarray, holistic: mp_holistic.Holistic):
    """
    Chạy inference Mediapipe trên frame và trả về kết quả.
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = holistic.process(rgb)
    rgb.flags.writeable = True
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr, results


def extract_keypoints(results) -> np.ndarray:
    """
    Chuyển kết quả Mediapipe thành vector keypoints phẳng.
    """
    pose_kps = np.zeros((N_UPPER_BODY_POSE_LANDMARKS, 3))
    left_hand_kps = np.zeros((N_HAND_LANDMARKS, 3))
    right_hand_kps = np.zeros((N_HAND_LANDMARKS, 3))

    if results and results.pose_landmarks:
        for i, lm in enumerate(results.pose_landmarks.landmark[:N_UPPER_BODY_POSE_LANDMARKS]):
            pose_kps[i] = [lm.x, lm.y, lm.z]

    if results and results.left_hand_landmarks:
        left_hand_kps = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark])

    if results and results.right_hand_landmarks:
        right_hand_kps = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark])

    keypoints = np.concatenate([pose_kps, left_hand_kps, right_hand_kps])
    return keypoints.flatten()


def sequence_frames(video_path: str, holistic: mp_holistic.Holistic) -> List[np.ndarray]:
    """
    Đọc video và lấy mẫu tối đa 60 frame đã chuyển thành keypoints.
    """
    frames: List[np.ndarray] = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Không thể mở video input")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // 60)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % step != 0:
            continue

        _, results = mediapipe_detection(frame, holistic)
        keypoints = extract_keypoints(results)
        frames.append(keypoints)

    cap.release()
    return frames


def interpolate_keypoints(sequence: List[np.ndarray], target_len: int = 60) -> np.ndarray:
    """
    Nội suy chuỗi keypoints về độ dài target_len (mặc định 60).
    """
    if not sequence:
        raise ValueError("Chuỗi keypoints trống")

    original_times = np.linspace(0, 1, len(sequence))
    target_times = np.linspace(0, 1, target_len)
    num_features = sequence[0].shape[0]

    interpolated = np.zeros((target_len, num_features))
    for idx in range(num_features):
        feature_values = [frame[idx] for frame in sequence]
        interpolator = interp1d(
            original_times,
            feature_values,
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate",
        )
        interpolated[:, idx] = interpolator(target_times)

    return interpolated


def predict_from_video(video_path: str) -> Dict[str, object]:
    """
    Pipeline suy luận từ file video, trả kết quả JSON-friendly.
    """
    model = load_model()
    _, inv_label_map = load_label_map()

    holistic = create_holistic()
    frames = sequence_frames(video_path, holistic)
    if not frames:
        raise ValueError("Không trích xuất được keypoints từ video")

    keypoints = interpolate_keypoints(frames)
    prediction = model.predict(np.expand_dims(keypoints, axis=0))
    idx = int(np.argmax(prediction, axis=1)[0])

    return {
        "label": inv_label_map[idx],
        "confidence": float(prediction[0][idx]),
        "probs": prediction[0].tolist(),
    }


def predict_from_bytes(video_bytes: bytes) -> Dict[str, object]:
    """
    Nhận video dạng bytes (ví dụ UploadFile trong FastAPI), chạy dự đoán.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        temp_path = tmp.name

    try:
        return predict_from_video(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
