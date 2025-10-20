import os
import time
import threading
from collections import deque
from datetime import datetime

import av
import cv2
import numpy as np
import joblib
import streamlit as st
from PIL import Image
from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    RTCConfiguration,
    WebRtcMode,
)
import torch
from transformers import AutoImageProcessor, AutoModel
from ultralytics import YOLO
from sklearn.preprocessing import MultiLabelBinarizer

# ---------------------------

# ========== CONFIG =========

# ---------------------------

# Konfigurasi Path
ABS_BASE = "D:\\Projects\\Streamlit DINO"
REL_BASE = "./models"
# Jika D:/Dino/ tidak ada, maka fallback ke ./models
FALLBACK_BASE = "D:/Dino/" if os.path.exists("D:\Projects\Streamlit DINO") else REL_BASE
BASE_PATH = ABS_BASE if os.path.exists(ABS_BASE) else FALLBACK_BASE

YOLO_MODEL_PATH = os.path.join(BASE_PATH, "best.pt")
MODEL_DIR = os.path.join(BASE_PATH, "OVR_Checkpoints-20251018T053026Z-1-001", "OVR_Checkpoints")
ENCODER_PATH = os.path.join(BASE_PATH, "dinov3_multilabel_encoder.pkl")
MAPPING_SAVE_PATH = os.path.join(BASE_PATH, "label_mapping_dict.joblib")

label_columns = ["product", "grade", "cap", "label", "brand", "type", "subtype", "volume"]

# OPTIMASI: Otomatis menggunakan GPU jika tersedia
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HISTORY_MAXLEN = 10
predictions_store = deque(maxlen=HISTORY_MAXLEN)
predictions_lock = threading.Lock()

# Warna dalam format BGR (OpenCV)
CATEGORY_COLORS_BGR = {
    "product": (0, 255, 209),    
    "grade": (108, 255, 0),
    "cap": (0, 207, 255),
    "label": (255, 209, 0),
    "brand": (122, 240, 255),
    "type": (208, 138, 255),
    "subtype": (255, 122, 205),
    "volume": (105, 255, 148)
}

DINOV3_MODEL_NAME = "facebook/dinov3-convnext-small-pretrain-lvd1689m"

# -----------------------------

# ========== STREAMLIT UI =====

# -----------------------------

st.set_page_config(page_title="EkoVision Live Dashboard", layout="wide")
st.markdown(
""" <style>
.stApp { background-color: #0b0f14; color: #e6f7ff; }
.block-container { padding-top: 1rem; }
.stSidebar { background-color: #071019; color:#aee9f7; }
.card {
    background: linear-gradient(180deg, rgba(10,20,24,0.9), rgba(3,9,12,0.8));
    border-radius: 10px;
    padding: 10px;
    margin-bottom: 8px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.6);
    color: #e7fbff;
}
.muted { color: #8fb7bf; font-size: 13px; } </style>
""",
unsafe_allow_html=True,
)

st.title(f"🎛️ EkoVision Live Dashboard — (Device: {device.type.upper()})")
st.caption("Real-time webcam · YOLOv10 + DINOv3 + multi-label classifiers")

# Sidebar settings

st.sidebar.header("Settings & Info")
st.sidebar.markdown(f"**Model base path**: `{BASE_PATH}`")
if device.type == 'cuda':
    st.sidebar.success("GPU (CUDA) is active! 🚀")
else:
    st.sidebar.warning("Using CPU. Performance may be slow.")

fps_limit = st.sidebar.slider("Max FPS (approx.)", min_value=1, max_value=12, value=6, step=1)
# Threshold untuk YOLO dan Probabilitas Fallback
confidence_threshold = st.sidebar.slider("Confidence Threshold (YOLO & Fallback)", 0.0, 1.0, 0.25, 0.05)
history_count = st.sidebar.slider("History entries to show", 1, HISTORY_MAXLEN, 5)

st.sidebar.markdown("---")
st.sidebar.markdown("Run (local network):")
st.sidebar.code("streamlit run app.py --server.address 0.0.0.0")
st.sidebar.markdown("---")
if st.sidebar.button("Clear history"):
    with predictions_lock:
        predictions_store.clear()

# -----------------------------

# ========== MODEL LOADER =====

# -----------------------------

@st.cache_resource(show_spinner=False)
def load_models_from_paths():
    if not os.path.exists(YOLO_MODEL_PATH):
        raise FileNotFoundError(f"YOLO weights not found at {YOLO_MODEL_PATH}")
    yolo_model = YOLO(YOLO_MODEL_PATH).to(device)

    dinov3_processor = AutoImageProcessor.from_pretrained(DINOV3_MODEL_NAME)
    dinov3_model = AutoModel.from_pretrained(DINOV3_MODEL_NAME).to(device).eval()

    if not os.path.exists(ENCODER_PATH) or not os.path.exists(MAPPING_SAVE_PATH):
        raise FileNotFoundError("Encoder or mapping file missing.")
        
    mlb = joblib.load(ENCODER_PATH)
    mapping_dict = joblib.load(MAPPING_SAVE_PATH)
    
    all_classifiers = []
    missing_count = 0
    
    num_classes = len(mlb.classes_)

    for i in range(num_classes):
        class_name = mlb.classes_[i]
        safe_name = str(class_name).replace(" ", "_").replace("/", "_").replace(":", "_").replace(".", "_")
        clf_path = os.path.join(MODEL_DIR, f"clf_{i}_{safe_name}.pkl")

        if not os.path.exists(clf_path):
            alt_path = os.path.join(MODEL_DIR, f"clf_{i}_{class_name}.pkl")
            if os.path.exists(alt_path):
                clf_path = alt_path
            else:
                st.warning(f"⚠️ Missing classifier: {clf_path} — skipped.")
                missing_count += 1
                continue

        try:
            all_classifiers.append(joblib.load(clf_path))
        except Exception as e:
            st.error(f"Error loading {clf_path}: {e}")
            continue

    st.sidebar.info(f"Loaded {len(all_classifiers)} classifiers, {missing_count} missing.")
    return yolo_model, dinov3_processor, dinov3_model, mlb, mapping_dict, all_classifiers

models_loaded = False
try:
    with st.spinner("Loading AI Models..."):
        yolo_model, dinov3_processor, dinov3_model, mlb, mapping_dict, all_classifiers = load_models_from_paths()
    models_loaded = True
    st.sidebar.success("Models loaded ✅")
except Exception as e:
    st.sidebar.error(f"Model load problem: {e}")
    st.error("Gagal memuat model. Periksa path model dan log di sidebar.")
    yolo_model = dinov3_processor = dinov3_model = mlb = mapping_dict = all_classifiers = None

# -----------------------------

# ========== PREDICT & LOGIC (Updated) ==========

# -----------------------------

def extract_dinov3_features(image_crop):
    inputs = dinov3_processor(images=image_crop, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = dinov3_model(**inputs)
        features = outputs.last_hidden_state[:, 0, :]
        return features.cpu().numpy().flatten()

def predict_frame_from_bgr(img_bgr):
    if not models_loaded:
        return img_bgr, {"info": "models not loaded"}

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    
    # Deteksi YOLOv10m
    results = yolo_model(img_pil, verbose=False, conf=confidence_threshold) 
    
    if not results or not results[0].boxes:
        return img_bgr, {"info": "No object detected"}

    boxes = results[0].boxes.cpu().numpy()
    best_box = boxes[np.argmax(boxes.conf)]
    yolo_conf = best_box.conf[0]
    x_min, y_min, x_max, y_max = map(int, best_box.xyxy[0])

    # Ekstraksi DINOv3
    image_crop = img_pil.crop((x_min, y_min, x_max, y_max))
    features = extract_dinov3_features(image_crop)

    # Klasifikasi (Probabilitas)
    X_pred = features.reshape(1, -1)
    Y_proba_list = [clf.predict_proba(X_pred)[0, 1] for clf in all_classifiers]
    Y_proba = np.array(Y_proba_list)
    
    class_proba_map = dict(zip(mlb.classes_, Y_proba))
    # Prediksi biner menggunakan threshold 0.5
    Y_pred_biner = (Y_proba > 0.5).astype(int).reshape(1, -1)
    predicted_labels_set = set(mlb.inverse_transform(Y_pred_biner)[0])
    
    final_output = {}
    
    # Logika Pemetaan & Fallback (Diambil dari script pengujian Anda)
    for col in label_columns:
        intersection = predicted_labels_set.intersection(mapping_dict[col])
        
        # Cari probabilitas tertinggi dalam kategori ini (untuk Fallback/Conflict)
        best_proba = -1
        best_label = "UNKNOWN"
        for label in mapping_dict[col]:
            proba = class_proba_map.get(label, 0.0)
            if proba > best_proba:
                best_proba = proba
                best_label = label
        
        if len(intersection) == 1:
            # 1. Tidak Ada Konflik, Prediksi Kuat
            final_output[col] = intersection.pop()
        
        elif len(intersection) == 0:
            # 2. Fallback jika tidak ada prediksi biner (di bawah 0.5)
            # Menggunakan confidence_threshold dari sidebar
            if best_proba >= confidence_threshold: 
                final_output[col] = f"{best_label} ({best_proba*100:.1f}%)"
            else:
                final_output[col] = "UNKNOWN"
        
        else:
            # 3. KONFLIK: Lebih dari satu prediksi biner (sangat jarang, pilih proba tertinggi)
            final_output[col] = f"CONFLICT -> {best_label} ({best_proba*100:.1f}%)"


    # Drawing BBOX dan Teks pada frame
    cv2.rectangle(img_bgr, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    
    # Posisi awal teks (sedikit di atas bounding box)
    y_text_start = y_min - 10 if y_min > 200 else y_max + 30 
    
    # Tampilkan Confidence Score YOLO
    yolo_conf_text = f"YOLO Conf: {yolo_conf:.2f}"
    cv2.putText(img_bgr, yolo_conf_text, (x_min, y_text_start - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2) # Warna Kuning
    
    # Tampilkan SEMUA 8 Atribut
    text_lines = [f"{k}: {v}" for k, v in final_output.items()]
    
    for i, line in enumerate(text_lines):
        # Ambil warna berdasarkan KATEGORI (k)
        color_bgr = CATEGORY_COLORS_BGR.get(k, (0, 255, 255))
        cv2.putText(img_bgr, line, (x_min, y_text_start + i * 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)

    return img_bgr, final_output

# -----------------------------

# ========== VIDEO CLASS ======

# -----------------------------

class StreamProcessor(VideoProcessorBase):
    def __init__(self):
        self._last_time = 0.0
        self._min_interval = 1.0 / max(1, fps_limit) 

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        now = time.time()
        if now - self._last_time < self._min_interval: 
            return frame
        self._last_time = now

        img_bgr = frame.to_ndarray(format="bgr24")
        out_bgr, final_out = predict_frame_from_bgr(img_bgr.copy())
        
        with predictions_lock:
            predictions_store.appendleft({
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "pred": final_out
            })
        return av.VideoFrame.from_ndarray(out_bgr, format="bgr24")

# -----------------------------

# ========== MAIN LAYOUT ======

# -----------------------------

col_left, col_right = st.columns([2, 1])

with col_left:
    st.markdown("### Live Camera")
    st.info("Start stream and allow camera access. Adjust 'Max FPS' in the sidebar if stream is laggy.")
    rtc_conf = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    webrtc_ctx = webrtc_streamer(
        key="ekovision",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_conf,
        video_processor_factory=StreamProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col_right:
    st.markdown("### Latest Prediction")
    if len(predictions_store) == 0:
        st.markdown("<div class='card'><span class='muted'>Waiting for object detection and classification...</span></div>", unsafe_allow_html=True)
    else:
        with predictions_lock:
            # Tampilkan hanya history_count entri terbaru
            for idx in range(min(history_count, len(predictions_store))):
                latest = predictions_store[idx]
                ts = latest["timestamp"]
                pred = latest["pred"]
                
                # Gunakan format card yang lebih menarik untuk history
                card_content = [f"**{col.capitalize()}**: {pred.get(col, 'N/A')}" for col in label_columns]
                
                # Tampilkan entri terbaru paling atas
                if idx == 0:
                    st.markdown(f"<div class='card'><strong>Latest — {ts}</strong><br>" + "<br>".join(card_content) + "</div>", unsafe_allow_html=True)
                # Tampilkan history di bawahnya
                elif idx == 1:
                    st.markdown("---")
                    st.markdown("##### History")
                    st.markdown(f"<div class='card'><span class='muted'>{ts}</span><br>" + "<br>".join(card_content) + "</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='card'><span class='muted'>{ts}</span><br>" + "<br>".join(card_content) + "</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.caption(f"⚙️ Running on **{device.type.upper()}**. *Label diikuti (xx.x%) berarti hasil fallback / conflict.*")