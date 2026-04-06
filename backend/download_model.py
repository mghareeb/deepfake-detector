"""Download model weights from Hugging Face Hub on first startup.

This script checks for cached weights and downloads them if missing.
It is called automatically by the Docker entrypoint, and can also be
run standalone: ``python download_model.py``

Environment variables
---------------------
HF_MODEL_REPO : str
    Hugging Face Hub repository ID (default: "your-username/deepfake-efficientnet-b4").
    Change this to point at your own fine-tuned checkpoint.
HF_MODEL_FILE : str
    Filename inside the repo (default: "efficientnet_b4_deepfake.pth").
HF_HOME : str
    Hugging Face cache root (default: /home/user/.cache/huggingface inside the
    container). Automatically set by the Dockerfile.
"""

import os
import sys
from pathlib import Path

# The weights are stored next to the application code so that model.py can
# locate them with a simple relative import.
WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
DEFAULT_REPO = os.environ.get("HF_MODEL_REPO", "your-username/deepfake-efficientnet-b4")
DEFAULT_FILE = os.environ.get("HF_MODEL_FILE", "efficientnet_b4_deepfake.pth")


def download_weights(
    repo_id: str = DEFAULT_REPO,
    filename: str = DEFAULT_FILE,
) -> Path:
    """Return the local path to the weight file, downloading if necessary."""
    dest = WEIGHTS_DIR / filename

    if dest.exists():
        print(f"[download_model] Weights already cached at {dest}")
        return dest

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Try Hugging Face Hub download ────────────────────────────────────
    try:
        from huggingface_hub import hf_hub_download

        print(f"[download_model] Downloading {filename} from {repo_id} ...")
        cached = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(WEIGHTS_DIR),
            local_dir_use_symlinks=False,
        )
        print(f"[download_model] Saved to {cached}")
        return Path(cached)

    except Exception as exc:
        print(
            f"[download_model] Could not download from Hub ({exc}). "
            "Falling back to ImageNet-pretrained backbone only."
        )

    # ── Fallback: create a sentinel so we don't retry every boot ─────────
    sentinel = WEIGHTS_DIR / ".no_finetuned_weights"
    sentinel.touch()
    print(f"[download_model] Sentinel written to {sentinel}")
    return sentinel


if __name__ == "__main__":
    path = download_weights()
    print(f"Weights path: {path}")
    sys.exit(0)
