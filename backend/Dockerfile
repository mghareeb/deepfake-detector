# ---- Deepfake Detector – Hugging Face Spaces Dockerfile ----
# Targets the free T4-small GPU tier.  Builds on the official
# CUDA-enabled Python base so that PyTorch can use the GPU out
# of the box.

FROM python:3.11-slim

# ── System deps ──────────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Non-root user (required by HF Spaces) ───────────────────
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /home/user/app

# ── Python dependencies ──────────────────────────────────────
# Install PyTorch with CUDA 12.1 support (matches the T4 runtime).
# Pinned to cu121 index so the download is ~800 MB instead of 2 GB.
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir \
        torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r requirements.txt

# ── Application code ─────────────────────────────────────────
COPY --chown=user:user . .

# ── Cache & weights directories ──────────────────────────────
RUN mkdir -p /home/user/.cache/huggingface weights && \
    chown -R user:user /home/user/.cache weights

ENV HF_HOME=/home/user/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/user/.cache/huggingface \
    HF_MODEL_REPO="your-username/deepfake-efficientnet-b4" \
    HF_MODEL_FILE="efficientnet_b4_deepfake.pth"

USER user

# ── HF Spaces expects port 7860 ─────────────────────────────
EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
