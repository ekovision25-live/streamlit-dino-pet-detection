import torch
import cv2
import numpy as np
import joblib
import os
import streamlit as st
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from ultralytics import YOLO

# ======================================================
# CONFIGURATION
# ======================================================
# Resolve paths relative to this file so it works locally and on Streamlit Cloud
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
YOLO_MODEL_PATH = os.path.join(BASE_PATH, "best.pt")
MODEL_DIR = os.path.join(BASE_PATH, "OVR_Checkpoints-20251018T053026Z-1-001", "OVR_Checkpoints")
ENCODER_PATH = os.path.join(BASE_PATH, "dinov3_multilabel_encoder.pkl")
MAPPING_SAVE_PATH = os.path.join(BASE_PATH, "label_mapping_dict.joblib")

label_columns = ["product", "grade", "cap", "label", "brand", "type", "subtype", "volume"]

device = torch.device("cpu")  # Using CPU
print(f"Using device: {device}")

# ======================================================
# Load YOLOv10m and DINOv3
# ======================================================
yolo_model = YOLO(YOLO_MODEL_PATH)
print("✅ YOLOv10m loaded.")

dinov3_model_name = "facebook/dinov3-convnext-small-pretrain-lvd1689m"
hf_cache_dir = os.path.join(BASE_PATH, ".hf_cache")
os.makedirs(hf_cache_dir, exist_ok=True)

# Optional: use Hugging Face token from Streamlit secrets (for Streamlit Cloud)
hf_token = None
try:
    if hasattr(st, "secrets"):
        hf_token = st.secrets.get("HF_TOKEN")
except Exception:
    hf_token = None

def _load_hf_vision_model(model_name):
    processor = AutoImageProcessor.from_pretrained(
        model_name,
        cache_dir=hf_cache_dir,
        token=hf_token,
        trust_remote_code=False
    )
    model = AutoModel.from_pretrained(
        model_name,
        cache_dir=hf_cache_dir,
        token=hf_token,
        trust_remote_code=False
    )
    return processor, model

try:
    dinov3_processor, dinov3_model = _load_hf_vision_model(dinov3_model_name)
    dinov3_model = dinov3_model.to(device).eval()
    print("✅ DINOv3 loaded.")
except Exception as e:
    st.error(
        "Failed to load DINOv3 backbone from Hugging Face. Please set an 'HF_TOKEN' in Streamlit "
        "secrets to increase rate limits, then redeploy."
    )
    raise

# ======================================================
# Load Classifiers
# ======================================================
mlb = joblib.load(ENCODER_PATH)
mapping_dict = joblib.load(MAPPING_SAVE_PATH)

num_classes = len(mlb.classes_)
all_classifiers = []

for i in range(num_classes):
    class_name = mlb.classes_[i]
    # Create safe filename by replacing problematic characters but keep the original structure
    safe_class_name = str(class_name).replace(' ', '_').replace('/', '_').replace(':', '_')
    # Don't replace dots as they might be part of filenames like batch0.jpg
    classifier_path = os.path.join(MODEL_DIR, f"clf_{i}_{safe_class_name}.pkl")

    if os.path.exists(classifier_path):
        all_classifiers.append(joblib.load(classifier_path))
    else:
        print(f"❌ ERROR: Model Class {class_name} missing. Check folder.")
        exit()

print(f"✅ {len(all_classifiers)} classifiers loaded.")

# Infer expected feature dimension from first classifier
EXPECTED_FEATURE_DIM = None
if all_classifiers:
    try:
        EXPECTED_FEATURE_DIM = int(all_classifiers[0].coef_.shape[1])
    except Exception:
        EXPECTED_FEATURE_DIM = None

# ======================================================
# Feature Extraction using DINOv3
# ======================================================
def extract_dinov3_features(image_crop):
    # Ensure image is in RGB format
    if image_crop.mode != 'RGB':
        image_crop = image_crop.convert('RGB')
    
    # Resize image to ensure consistent input size
    image_crop = image_crop.resize((224, 224))
    
    inputs = dinov3_processor(images=image_crop, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = dinov3_model(**inputs)
        cls_token_features = outputs.last_hidden_state[:, 0, :]
    feats = cls_token_features.cpu().numpy().flatten()
    if EXPECTED_FEATURE_DIM is not None and feats.shape[0] != EXPECTED_FEATURE_DIM:
        st.error(
            f"Backbone feature size {feats.shape[0]} does not match classifier expectation "
            f"{EXPECTED_FEATURE_DIM}. Ensure DINOv3 backbone is downloaded correctly."
        )
        raise ValueError("Feature dimension mismatch")
    return feats

# ======================================================
# Real-time Prediction Logic
# ======================================================
def predict_frame(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    
    # YOLO detection
    results = yolo_model(img_pil, verbose=False)
    
    if not results or not results[0].boxes:
        # Return empty classification results when no object is detected
        empty_results = {col: "No detection" for col in label_columns}
        return img_bgr, empty_results
    
    # Get the best bounding box
    boxes = results[0].boxes.cpu().numpy()
    best_box = boxes[np.argmax(boxes.conf)]
    x_min, y_min, x_max, y_max = map(int, best_box.xyxy[0])

    # Crop and Extract features using DINOv3
    image_crop = img_pil.crop((x_min, y_min, x_max, y_max))
    features = extract_dinov3_features(image_crop)

    # Classification (Probabilities)
    X_pred = features.reshape(1, -1)
    Y_proba_list = [clf.predict_proba(X_pred)[0, 1] for clf in all_classifiers]
    Y_proba = np.array(Y_proba_list)

    class_proba_map = dict(zip(mlb.classes_, Y_proba)) 
    
    # Debug: Print top predictions for each category
    print("=== DEBUG CLASSIFICATION ===")
    for col in label_columns:
        col_labels = mapping_dict[col]
        col_probas = [(label, class_proba_map.get(label, 0.0)) for label in col_labels]
        col_probas.sort(key=lambda x: x[1], reverse=True)
        print(f"\n{col.upper()} - Top 5 predictions:")
        for i, (label, proba) in enumerate(col_probas[:5]):
            print(f"  {i+1}. {label}: {proba*100:.2f}%")
    
    # Use a more sophisticated classification approach
    # First, get the best prediction for each category based on probability
    final_output = {}
    
    for col in label_columns:
        col_labels = mapping_dict[col]
        col_probas = [(label, class_proba_map.get(label, 0.0)) for label in col_labels]
        col_probas.sort(key=lambda x: x[1], reverse=True)
        
        best_label, best_proba = col_probas[0]
        
        # Use high confidence thresholds (75-80% minimum)
        if col == 'product':
            threshold = 0.75  # 75% minimum for product classification
        elif col in ['brand', 'type']:
            threshold = 0.70  # 70% minimum for important categories
        else:
            threshold = 0.65  # 65% minimum for other categories
        
        if best_proba > threshold:
            if best_proba > 0.90:
                final_output[col] = best_label  # Very high confidence, no percentage shown
            else:
                final_output[col] = f"{best_label} ({best_proba*100:.1f}%)"
        else:
            # If no good prediction, try the second best
            if len(col_probas) > 1:
                second_best_label, second_best_proba = col_probas[1]
                if second_best_proba > threshold * 0.9:  # 90% of the threshold
                    final_output[col] = f"{second_best_label} ({second_best_proba*100:.1f}%)"
                else:
                    # Instead of "UNKNOWN", choose the best available label with low confidence indicator
                    final_output[col] = f"{best_label} ({best_proba*100:.1f}%) [Low Confidence]"
            else:
                # Instead of "UNKNOWN", choose the best available label with low confidence indicator
                final_output[col] = f"{best_label} ({best_proba*100:.1f}%) [Low Confidence]"

    # Draw Bounding Box and Add Text to Frame
    cv2.rectangle(img_bgr, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    y_text = y_min - 10 if y_min > 20 else y_max + 30

    # Add all label attributes (8 fields)
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
    
    for i, line in enumerate(text_lines):
        cv2.putText(img_bgr, line, (x_min, y_text + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return img_bgr, final_output

# ======================================================
# Streamlit App Setup
# ======================================================
def main():
    st.set_page_config(
        page_title="Ekovision PET Detection Model",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Ekovision PET Detection Model")
    st.markdown("Detect and classify PET bottles using YOLO and DINOv3")
    
    # Add tabs for different modes
    tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Live Camera"])
    
    with tab1:
        st.subheader("Upload an Image")
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file", 
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image to detect and classify objects"
        )
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Process the image
            with st.spinner("Processing image..."):
                # Convert PIL image to numpy array for processing
                img_array = np.array(image)
                processed_frame, prediction_data = predict_frame(img_array)
            
            # Display results
            st.success("✅ Image processed successfully!")
            
            # Create two columns for results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Classification Results")
                for key, value in prediction_data.items():
                    st.write(f"**{key.title()}:** {value}")
            
            with col2:
                st.subheader("🖼️ Processed Image")
                # Convert BGR to RGB for display
                processed_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                st.image(processed_rgb, caption="Image with Detection", use_container_width=True)
        
        else:
            st.info("👆 Please upload an image to get started")
    
    with tab2:
        st.subheader("Live Camera Detection")
        st.info("Click 'Start Camera' to begin real-time detection")
        
        # Camera controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            start_camera = st.button("📷 Start Camera", type="primary")
        
        with col2:
            stop_camera = st.button("⏹️ Stop Camera")
        
        with col3:
            if 'camera_running' not in st.session_state:
                st.session_state.camera_running = False
        
        # Camera placeholder
        camera_placeholder = st.empty()
        results_placeholder = st.empty()
        
        if start_camera:
            st.session_state.camera_running = True
            st.success("Camera started! Detection will begin shortly...")
        
        if stop_camera:
            st.session_state.camera_running = False
            st.info("Camera stopped.")
        
        # Camera functionality
        if st.session_state.camera_running:
            # Use Streamlit's camera input as a more reliable alternative
            camera_feed = st.camera_input("Live Camera Feed")
            
            if camera_feed is not None:
                try:
                    # Convert camera input to numpy array
                    img_array = np.array(Image.open(camera_feed))
                    
                    # Process the frame
                    processed_frame, prediction_data = predict_frame(img_array)
                    
                    # Ensure prediction_data is a dictionary
                    if not isinstance(prediction_data, dict):
                        prediction_data = {col: "Processing..." for col in label_columns}
                    
                    # Display processed frame
                    processed_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    camera_placeholder.image(processed_rgb, caption="Live Detection", use_container_width=True)
                    
                    # Display classification results
                    with results_placeholder.container():
                        st.subheader("📊 Live Classification Results")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            for i, (key, value) in enumerate(list(prediction_data.items())[:4]):
                                st.write(f"**{key.title()}:** {value}")
                        
                        with col2:
                            for i, (key, value) in enumerate(list(prediction_data.items())[4:]):
                                st.write(f"**{key.title()}:** {value}")
                
                except Exception as e:
                    st.error(f"❌ Processing error: {str(e)}")
                    camera_placeholder.image(camera_feed, caption="Live Camera Feed", use_container_width=True)
            
            else:
                st.info("📷 Camera is ready. Please allow camera access when prompted.")

if __name__ == '__main__':
    main()
