"""Pre-download HuggingFace models on startup so first request isn't slow.

The transformers library caches models automatically under HF_HOME.
This script just triggers that download eagerly.
"""

import os
import sys


HF_MODEL_1 = os.environ.get(
    "HF_DEEPFAKE_MODEL", "dima806/deepfake_vs_real_image_detection"
)
HF_MODEL_2 = "buildborderless/CommunityForensics-DeepfakeDet-ViT"


def download_weights() -> None:
    """Pre-download both models and processors from HuggingFace Hub."""
    try:
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        for model_id in (HF_MODEL_1, HF_MODEL_2):
            print(f"[download_model] Ensuring model cached: {model_id}")
            AutoImageProcessor.from_pretrained(model_id, trust_remote_code=True)
            AutoModelForImageClassification.from_pretrained(model_id, trust_remote_code=True)
            print(f"[download_model] Model ready: {model_id}")

    except Exception as exc:
        print(
            f"[download_model] Could not download model ({exc}). "
            "It will be downloaded on first request."
        )


if __name__ == "__main__":
    download_weights()
    sys.exit(0)
