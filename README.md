# Ekovision PET Detection Model

A Streamlit application that detects and classifies PET bottles using YOLO and DINOv3 models.

## Features

- **Image Upload**: Upload images for detection and classification
- **Live Camera**: Real-time detection using webcam
- **Multi-label Classification**: Classifies products by:
  - Product type
  - Grade
  - Cap presence
  - Label presence
  - Brand
  - Type
  - Subtype
  - Volume

## Usage

1. Upload an image or use live camera
2. The app will detect objects and classify them
3. View detailed classification results

## Models Used

- YOLOv10m for object detection
- DINOv3 for feature extraction
- Custom classifiers for multi-label classification
