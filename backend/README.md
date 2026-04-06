---
title: Deepfake Detector
emoji:🛡️
colorFrom: indigo
colorTo: cyan
sdk: docker
app_port: 7860
suggested_hardware: t4-small
suggested_storage: small
pinned: false
license: mit
---

# Deepfake Detector API

EfficientNet-B4 + FFT frequency-analysis ensemble for deepfake image detection,
served as a FastAPI backend with Grad-CAM heatmap visualisation.

## Endpoints

| Method | Path       | Description                                  |
|--------|-----------|----------------------------------------------|
| GET    | `/health` | Server and model status                      |
| POST   | `/predict`| Upload an image, returns fake score + heatmap|

## Custom weights

Set the `HF_MODEL_REPO` and `HF_MODEL_FILE` environment variables in your
Space settings to point at your own fine-tuned checkpoint on the Hub.
Weights are downloaded automatically on first startup.
