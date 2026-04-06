# Deepfake Detector

AI-powered deepfake image detection using EfficientNet-B4 with Grad-CAM visualization.

![Demo Screenshot](docs/demo.png)
<!-- Replace with actual screenshots after running the app -->

## Architecture

```
Browser (React + Vite)
    |
    |  POST /predict (multipart image)
    v
FastAPI Backend
    |
    ├── Face Detection (OpenCV Haar Cascade)
    ├── EfficientNet-B4 Inference (PyTorch)
    └── Grad-CAM Heatmap Generation
    |
    v
JSON Response { fake_score, confidence, heatmap_base64 }
```

## Features

- Drag-and-drop image upload
- Real/Fake classification with confidence score
- Animated score meter (0-100%)
- Grad-CAM heatmap overlay showing which regions influenced the prediction
- Confidence breakdown bar chart
- Benchmark metrics display (AUC, F1, Accuracy)
- Face detection with automatic cropping

## Prerequisites

- Python 3.9+
- Node.js 18+
- npm 9+

## Setup

### Backend

```bash
cd backend

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For CPU-only PyTorch (smaller download):
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# pip install -r requirements.txt

# Start the server
uvicorn main:app --reload --port 8000
```

The first startup will download EfficientNet-B4 pretrained weights (~75 MB).

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 in your browser.

## API

### `POST /predict`

Upload an image for deepfake analysis.

**Request:** `multipart/form-data` with a `file` field (JPEG, PNG, or WebP, max 10 MB)

**Response:**
```json
{
  "fake_score": 0.73,
  "real_score": 0.27,
  "confidence": 0.73,
  "grad_cam_heatmap": "<base64 PNG>",
  "face_detected": true,
  "message": "Image analyzed successfully. A face was detected and used for analysis."
}
```

### `GET /health`

Returns server and model status.

## Model Notes

This project uses **EfficientNet-B4** architecture with **ImageNet pretrained weights**. The binary classification head (real vs fake) is randomly initialized.

**For meaningful deepfake detection**, the model needs to be fine-tuned on a deepfake dataset such as:
- [FaceForensics++](https://github.com/ondyari/FaceForensics)
- [Celeb-DF](https://github.com/yuezunli/celeb-deepfakeforensics)
- [DFDC (Deepfake Detection Challenge)](https://ai.meta.com/datasets/dfdc/)

To use custom weights, save your fine-tuned state dict and load it in `backend/model.py`:
```python
model.load_state_dict(torch.load("path/to/weights.pth", map_location=device))
```

## Grad-CAM

The Grad-CAM visualization targets `model._conv_head` (the final convolutional layer of EfficientNet-B4). It highlights image regions that most strongly influenced the model's prediction, helping users understand *why* an image was classified as real or fake.

## Project Structure

```
deepfake-detector/
├── README.md
├── backend/
│   ├── requirements.txt
│   ├── main.py              # FastAPI app with /predict endpoint
│   ├── model.py              # EfficientNet-B4 loading and inference
│   ├── gradcam.py            # Grad-CAM heatmap generation
│   └── preprocessing.py      # Image loading, face detection, transforms
└── frontend/
    ├── package.json
    ├── vite.config.js
    ├── index.html
    └── src/
        ├── main.jsx
        ├── App.jsx
        ├── App.css
        ├── index.css
        ├── api/
        │   └── predict.js
        └── components/
            ├── DropZone.jsx / .css
            ├── ScoreMeter.jsx / .css
            ├── HeatmapOverlay.jsx / .css
            ├── ConfidenceChart.jsx / .css
            └── BenchmarkMetrics.jsx / .css
```

## Future Improvements

- Fine-tune on FaceForensics++ dataset
- Replace Haar cascade with MTCNN or RetinaFace for better face detection
- Add video frame-by-frame analysis
- Batch image processing
- Model versioning and A/B testing
- Docker deployment configuration
