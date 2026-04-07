"""Pre-download HuggingFace model on startup so first request isn't slow.

The transformers library caches models automatically under HF_HOME.
This script just triggers that download eagerly.

Environment variables
---------------------
HF_DEEPFAKE_MODEL : str
    HuggingFace Hub model ID (default: "Wvolf/ViT_Deepfake_Detection").
HF_HOME : str
    Hugging Face cache root (default: ~/.cache/huggingface).
"""

import os
import sys


HF_MODEL_ID = os.environ.get(
    "HF_DEEPFAKE_MODEL", "prithivMLmods/Deep-Fake-Detector-Model"
)


def download_weights() -> None:
    """Pre-download model and processor from HuggingFace Hub."""
    try:
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        print(f"[download_model] Ensuring model cached: {HF_MODEL_ID}")
        AutoImageProcessor.from_pretrained(HF_MODEL_ID)
        AutoModelForImageClassification.from_pretrained(HF_MODEL_ID)
        print(f"[download_model] Model ready: {HF_MODEL_ID}")

    except Exception as exc:
        print(
            f"[download_model] Could not download model ({exc}). "
            "It will be downloaded on first request."
        )


if __name__ == "__main__":
    download_weights()
    sys.exit(0)
