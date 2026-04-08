# ---- Deepfake Detector – Hugging Face Spaces Dockerfile ----
# Targets the free CPU Basic tier.  Uses CPU-only PyTorch to
# keep the image small (~1.2 GB vs ~2.5 GB with CUDA).

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
# Install CPU-only PyTorch (~200 MB vs ~800 MB for CUDA build).
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir \
        torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# ── Application code ─────────────────────────────────────────
COPY --chown=user:user . .

# ── Cache & weights directories ──────────────────────────────
RUN mkdir -p /home/user/.cache/huggingface weights && \
    chown -R user:user /home/user/.cache weights

ENV HF_HOME=/home/user/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/user/.cache/huggingface \
    HF_DEEPFAKE_MODEL="umm-maybe/AI-image-detector"

USER user

# ── HF Spaces expects port 7860 ─────────────────────────────
EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
