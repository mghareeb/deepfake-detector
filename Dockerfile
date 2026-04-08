# ---- Deepfake Detector – Hugging Face Spaces Dockerfile ----
# Multi-stage: build React frontend, then serve everything from FastAPI.

# ── Stage 1: build frontend ────────────────────────────────────
FROM node:20-slim AS frontend-build
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install
# v2: logo, favicon, badge text, image crop fix
COPY frontend/ .
RUN npm run build

# ── Stage 2: backend + static frontend ─────────────────────────
FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user (required by HF Spaces)
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /home/user/app

# Python dependencies — CPU-only PyTorch
COPY --chown=user:user backend/requirements.txt .
RUN pip install --no-cache-dir \
        torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Backend code
COPY --chown=user:user backend/ .

# Built frontend → static/
COPY --from=frontend-build --chown=user:user /app/dist ./static

# Cache directories
RUN mkdir -p /home/user/.cache/huggingface && \
    chown -R user:user /home/user/.cache

ENV HF_HOME=/home/user/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/user/.cache/huggingface \
    HF_DEEPFAKE_MODEL="buildborderless/CommunityForensics-DeepfakeDet-ViT"

USER user

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
