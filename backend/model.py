from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INPUT_SIZE = 380
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
FACE_MARGIN = 0.3
PIXEL_WEIGHT = 0.7
FREQ_WEIGHT = 0.3

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
WEIGHTS_FILE = os.environ.get("HF_MODEL_FILE", "efficientnet_b4_deepfake.pth")

_face_cascade = None
_model_state: dict | None = None  # singleton populated by load_model()

# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def _get_face_cascade():
    global _face_cascade
    if _face_cascade is None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(path)
    return _face_cascade


def _detect_and_crop_face(image_np: np.ndarray) -> tuple[np.ndarray, bool]:
    cascade = _get_face_cascade()
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    if len(faces) == 0:
        return image_np, False

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    img_h, img_w = image_np.shape[:2]
    mx, my = int(w * FACE_MARGIN), int(h * FACE_MARGIN)
    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(img_w, x + w + mx)
    y2 = min(img_h, y + h + my)
    return image_np[y1:y2, x1:x2], True


def _preprocess_tensor(image_np: np.ndarray) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transform(image_np).unsqueeze(0)


# ---------------------------------------------------------------------------
# Custom classifier head
# ---------------------------------------------------------------------------

class DeepfakeClassifier(nn.Module):
    """EfficientNet-B4 backbone with a custom binary classification head."""

    def __init__(self):
        super().__init__()
        base = EfficientNet.from_pretrained("efficientnet-b4")
        self.features = base  # full backbone up through _conv_head / _bn1
        in_features = base._fc.in_features  # 1792 for B4
        # Replace the stock FC with a custom head
        self.features._fc = nn.Identity()
        self.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)  # -> (B, 1792)
        return self.head(x)   # -> (B, 2)


# ---------------------------------------------------------------------------
# Grad-CAM (targets features._conv_head — last conv layer)
# ---------------------------------------------------------------------------

class _GradCAM:
    def __init__(self, model: DeepfakeClassifier):
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handles = [
            model.features._conv_head.register_forward_hook(self._fwd),
            model.features._conv_head.register_full_backward_hook(self._bwd),
        ]

    def _fwd(self, _module, _input, output):
        self.activations = output.detach()

    def _bwd(self, _module, _grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        original_image: np.ndarray,
        target_class: int = 1,
    ) -> str:
        """Return a base64-encoded PNG of the Grad-CAM heatmap overlaid on the image."""
        self.model.eval()
        inp = input_tensor.clone().requires_grad_(True)

        with torch.enable_grad():
            logits = self.model(inp)
            self.model.zero_grad()
            logits[0, target_class].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam /= cam.max()

        h, w = original_image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = (0.5 * heatmap + 0.5 * original_image).astype(np.uint8)

        _, buf = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        return base64.b64encode(buf).decode("utf-8")

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ---------------------------------------------------------------------------
# Frequency analysis (numpy FFT)
# ---------------------------------------------------------------------------

def _frequency_score(image_np: np.ndarray) -> float:
    """Analyse high-frequency energy via 2-D FFT.

    Deepfake images often exhibit subtle high-frequency artefacts from the
    generation process.  We compute the ratio of high-frequency energy to
    total energy in the magnitude spectrum — a higher ratio is more
    suspicious.  The raw ratio is passed through a sigmoid to map it into
    (0, 1) where 1 = likely fake.
    """
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
    # 2-D DFT, shift DC to centre
    dft = np.fft.fft2(gray)
    shifted = np.fft.fftshift(dft)
    magnitude = np.abs(shifted)

    rows, cols = gray.shape
    cy, cx = rows // 2, cols // 2
    # Radius threshold: frequencies beyond 30 % of the Nyquist distance
    radius = 0.3 * min(cy, cx)

    y_coords, x_coords = np.ogrid[:rows, :cols]
    dist = np.sqrt((y_coords - cy) ** 2 + (x_coords - cx) ** 2)

    total_energy = magnitude.sum() + 1e-8
    high_freq_energy = magnitude[dist > radius].sum()
    ratio = high_freq_energy / total_energy

    # Sigmoid centred at 0.55 — tuneable once real training data is available
    score = 1.0 / (1.0 + np.exp(-20 * (ratio - 0.55)))
    return float(np.clip(score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Model lifecycle
# ---------------------------------------------------------------------------

def load_model() -> dict:
    """Initialise the model, Grad-CAM, and device. Returns a state dict.

    If fine-tuned weights exist in ``weights/``, they are loaded into the
    classifier.  Otherwise the model runs with ImageNet-pretrained backbone
    and a randomly-initialised head (useful for development).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepfakeClassifier()

    weights_path = WEIGHTS_DIR / WEIGHTS_FILE
    if weights_path.exists():
        print(f"[model] Loading fine-tuned weights from {weights_path}")
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
    else:
        print("[model] No fine-tuned weights found — using ImageNet backbone only")

    model.eval()
    model.to(device)
    gradcam = _GradCAM(model)

    global _model_state
    _model_state = {"model": model, "device": device, "gradcam": gradcam}
    return _model_state


def cleanup():
    """Remove Grad-CAM hooks (call on shutdown)."""
    if _model_state is not None:
        _model_state["gradcam"].remove_hooks()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict(image_path: str) -> dict:
    """Run the full detection pipeline on an image file.

    Parameters
    ----------
    image_path : str
        Path to a JPEG / PNG / WebP image **or** raw bytes (when called
        from the FastAPI endpoint we also accept bytes via
        ``predict_from_bytes``).

    Returns
    -------
    dict with keys:
        score        – float 0-1, ensemble fake probability
        heatmap_base64 – base64 PNG of the Grad-CAM overlay
        freq_score   – float 0-1, frequency-domain fake probability
        pixel_score  – float 0-1, EfficientNet pixel-domain fake probability
        confidence   – float 0-1, max(real, fake) of ensemble
        face_detected – bool
    """
    image = Image.open(image_path).convert("RGB")
    return _run_pipeline(np.array(image))


def predict_from_bytes(image_bytes: bytes) -> dict:
    """Same as ``predict`` but accepts raw file bytes (used by the API)."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return _run_pipeline(np.array(image))


def _run_pipeline(image_np: np.ndarray) -> dict:
    state = _model_state
    if state is None:
        raise RuntimeError("Model not loaded — call load_model() first")

    model: DeepfakeClassifier = state["model"]
    device: torch.device = state["device"]
    gradcam: _GradCAM = state["gradcam"]

    # --- face detection & crop ------------------------------------------------
    cropped, face_detected = _detect_and_crop_face(image_np)
    display_image = cv2.resize(cropped, (INPUT_SIZE, INPUT_SIZE))
    tensor = _preprocess_tensor(cropped).to(device)

    # --- pixel-domain score (EfficientNet) ------------------------------------
    with torch.no_grad():
        logits = model(tensor)
    probs = F.softmax(logits, dim=1)[0]
    pixel_score = float(probs[1].item())  # P(fake)

    # --- Grad-CAM heatmap -----------------------------------------------------
    heatmap_b64 = gradcam.generate(tensor, display_image, target_class=1)

    # --- frequency-domain score -----------------------------------------------
    freq_score = _frequency_score(display_image)

    # --- weighted ensemble ----------------------------------------------------
    score = PIXEL_WEIGHT * pixel_score + FREQ_WEIGHT * freq_score
    score = float(np.clip(score, 0.0, 1.0))
    confidence = max(score, 1.0 - score)

    return {
        "score": round(score, 4),
        "heatmap_base64": heatmap_b64,
        "freq_score": round(freq_score, 4),
        "pixel_score": round(pixel_score, 4),
        "confidence": round(confidence, 4),
        "face_detected": face_detected,
    }
