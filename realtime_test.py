import torch
import cv2
import numpy as np
import joblib
import os
import pandas as pd
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from ultralytics import YOLO
from sklearn.preprocessing import MultiLabelBinarizer
from io import BytesIO

# ======================================================
# KONFIGURASI LOKAL (SESUAIKAN PATH INI DI PC ANDA)
# ======================================================
BASE_PATH = "D:/Dino/" # Pastikan menggunakan forward slash (/)
YOLO_MODEL_PATH = os.path.join(BASE_PATH, "best.pt")
# Pastikan nama folder checkpoint sudah benar
MODEL_DIR = os.path.join(BASE_PATH, "OVR_Checkpoints-20251018T053026Z-1-001", "OVR_Checkpoints") 
ENCODER_PATH = os.path.join(BASE_PATH, "dinov3_multilabel_encoder.pkl")
MAPPING_SAVE_PATH = os.path.join(BASE_PATH, "label_mapping_dict.joblib")

# Label columns (Harus sama dengan yang digunakan saat training)
label_columns = ["product", "grade", "cap", "label", "brand", "type", "subtype", "volume"]

# --- SETUP DEVICE & MODEL (CPU Lokal) ---
# Kita menggunakan CPU karena konflik driver CUDA
device = torch.device("cpu") 
print(f"Menggunakan perangkat: {device}")

# Load YOLO
try:
    yolo_model = YOLO(YOLO_MODEL_PATH)
    print("✅ YOLOv10m dimuat.")
except Exception as e:
    print(f"❌ GAGAL memuat YOLO: {e}")
    exit()

# Load DINOv3
dinov3_model_name = "facebook/dinov3-convnext-small-pretrain-lvd1689m"
dinov3_processor = AutoImageProcessor.from_pretrained(dinov3_model_name)
dinov3_model = AutoModel.from_pretrained(dinov3_model_name).to(device).eval()
print("✅ DINOv3 dimuat.")

# ======================================================
# FUNGSI EKSTRAKSI FITUR (DINOv3)
# ======================================================
def extract_dinov3_features(image_crop):
    inputs = dinov3_processor(images=image_crop, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = dinov3_model(**inputs)
        cls_token_features = outputs.last_hidden_state[:, 0, :]
    return cls_token_features.cpu().numpy().flatten()

# ======================================================
# MUAT SEMUA ASET CLASSIFIER (314 MODEL)
# ======================================================
try:
    mlb = joblib.load(ENCODER_PATH)
    mapping_dict = joblib.load(MAPPING_SAVE_PATH)
    
    num_classes = len(mlb.classes_)
    all_classifiers = []
    
    for i in range(num_classes):
        class_name = mlb.classes_[i]
        safe_class_name = str(class_name).replace(' ', '_').replace('/', '_').replace(':', '_').replace('.', '_')
        classifier_path = os.path.join(MODEL_DIR, f"clf_{i}_{safe_class_name}.pkl")
        
        # Logika fallback untuk nama file
        if not os.path.exists(classifier_path):
             classifier_path = os.path.join(MODEL_DIR, f"clf_{i}_{class_name}.pkl")
        
        if os.path.exists(classifier_path):
            all_classifiers.append(joblib.load(classifier_path))
        else:
            print(f"❌ ERROR KRITIS: Model Class {class_name} ({classifier_path}) hilang. Cek folder OVR_Checkpoints.")
            exit()
            
    print(f"✅ {len(all_classifiers)} Classifiers dimuat dan siap.")

except Exception as e:
    print(f"❌ GAGAL memuat aset Scikit-learn/Mapping: {e}")
    exit()

# ======================================================
# FUNGSI PREDIKSI REAL-TIME (INTI)
# ======================================================
def predict_frame(img_bgr):
    # Konversi BGR (OpenCV) ke RGB (YOLO/DINO/PIL)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    
    # Deteksi YOLOv10m
    results = yolo_model(img_pil, verbose=False)

    if not results or not results[0].boxes:
        return img_bgr, "No object detected."
    
    # Ambil BBOX terbaik
    boxes = results[0].boxes.cpu().numpy()
    best_box = boxes[np.argmax(boxes.conf)]
    x_min, y_min, x_max, y_max = map(int, best_box.xyxy[0])

    # Crop dan Ekstraksi DINOv3
    image_crop = img_pil.crop((x_min, y_min, x_max, y_max))
    features = extract_dinov3_features(image_crop)

    # 1. Klasifikasi (Probabilitas)
    X_pred = features.reshape(1, -1)
    Y_proba_list = [clf.predict_proba(X_pred)[0, 1] for clf in all_classifiers]
    Y_proba = np.array(Y_proba_list)
    
    class_proba_map = dict(zip(mlb.classes_, Y_proba)) 
    Y_pred_biner = (Y_proba > 0.5).astype(int).reshape(1, -1)
    predicted_labels_set = set(mlb.inverse_transform(Y_pred_biner)[0])
    
    final_output = {}
    
    # 2. Logika Pemetaan & Fallback
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
            final_output[col] = intersection.pop()
        
        elif len(intersection) == 0:
            # Fallback jika tidak ada prediksi biner
            if best_proba > 0.20:
                final_output[col] = f"{best_label} ({best_proba*100:.1f}%)"
            else:
                final_output[col] = "UNKNOWN"
        
        else:
            # KONFLIK: Pilih label proba tertinggi
            final_output[col] = f"CONFLICT -> {best_label} ({best_proba*100:.1f}%)"


    # Gambar BBOX dan Teks pada frame
    cv2.rectangle(img_bgr, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    
    # Format teks output (semua 8 atribut)
    y_text = y_min - 10 if y_min > 20 else y_max + 30
    
    # --- PERUBAHAN DI SINI: Menyertakan SEMUA 8 Atribut ---
    text_lines = [
        f"Product: {final_output.get('product', 'N/A')}",
        f"Grade: {final_output.get('grade', 'N/A')}",
        f"Cap: {final_output.get('cap', 'N/A')}",
        f"Label: {final_output.get('label', 'N/A')}",
        f"Brand: {final_output.get('brand', 'N/A')}",
        f"Type: {final_output.get('type', 'N/A')}",
        f"Subtype: {final_output.get('subtype', 'N/A')}",
        f"Volume: {final_output.get('volume', 'N/A')}"
    ]
    # -----------------------------------------------------
    
    for i, line in enumerate(text_lines):
        # Gunakan warna kuning-cyan (0, 255, 255) yang cerah untuk teks
        cv2.putText(img_bgr, line, (x_min, y_text + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return img_bgr, final_output

# ======================================================
# MAIN LOOP UNTUK KAMERA
# ======================================================
def start_realtime_test():
    cap = cv2.VideoCapture(0)  

    if not cap.isOpened():
        print("❌ GAGAL: Tidak dapat membuka kamera. Pastikan kamera terhubung.")
        return

    print("\n>>> UJI REAL-TIME DIMULAI. Tekan 'q' untuk keluar. <<<")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            processed_frame, prediction_data = predict_frame(frame)
        except Exception as e:
            # Menampilkan error di terminal, tapi tetap menampilkan frame
            print(f"Error prediksi: {e}") 
            processed_frame = frame
        
        # Tampilkan frame yang diproses
        cv2.imshow('EkoVision Real-time Test (Press Q to Quit)', processed_frame)
        
        # Logika keluar dari loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(">>> UJI REAL-TIME SELESAI. <<<")

# Jalankan fungsi utama
start_realtime_test()