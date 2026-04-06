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
# Default ensemble weights (no face detected — full image)
PIXEL_WEIGHT_DEFAULT = 0.7
FREQ_WEIGHT_DEFAULT = 0.3

# Face-specific ensemble weights (face detected — GAN artifacts more visible in freq domain)
PIXEL_WEIGHT_FACE = 0.6
FREQ_WEIGHT_FACE = 0.4

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
WEIGHTS_FILE = os.environ.get("HF_MODEL_FILE", "efficientnet_b4_deepfake.pth")

_face_cascade = None
_landmark_detector = None
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


def _get_landmark_detector():
    """Lazy-load OpenCV's LBF face landmark detector (68 points)."""
    global _landmark_detector
    if _landmark_detector is None:
        try:
            _landmark_detector = cv2.face.createFacemarkLBF()
            model_path = os.path.join(
                os.path.dirname(cv2.__file__),
                "data",
                "lbfmodel.yaml",
            )
            if os.path.exists(model_path):
                _landmark_detector.loadModel(model_path)
            else:
                # Download the model on first use
                import urllib.request
                url = "https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml"
                cache_dir = Path(__file__).resolve().parent / "weights"
                cache_dir.mkdir(parents=True, exist_ok=True)
                cached_model = cache_dir / "lbfmodel.yaml"
                if not cached_model.exists():
                    print("[model] Downloading LBF landmark model...")
                    urllib.request.urlretrieve(url, str(cached_model))
                _landmark_detector.loadModel(str(cached_model))
        except (AttributeError, Exception) as exc:
            # cv2.face module may not be available in all builds
            print(f"[model] Landmark detector unavailable: {exc}")
            _landmark_detector = "unavailable"
    return _landmark_detector if _landmark_detector != "unavailable" else None


def _landmark_consistency_score(image_np: np.ndarray) -> float:
    """Check face landmark symmetry and spacing consistency.

    GAN-generated faces often have subtle asymmetries in eye spacing,
    nose-to-mouth ratios, and jawline contours that deviate from
    natural facial geometry.  Returns 0.0 (perfectly consistent) to
    1.0 (highly inconsistent → likely fake).
    """
    detector = _get_landmark_detector()
    if detector is None:
        return _landmark_consistency_fallback(image_np)

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    cascade = _get_face_cascade()
    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

    if len(faces) == 0:
        return 0.0

    ok, landmarks = detector.fit(gray, faces)
    if not ok or len(landmarks) == 0:
        return _landmark_consistency_fallback(image_np)

    pts = landmarks[0][0]  # 68 landmarks, shape (68, 2)
    return _analyse_landmark_geometry(pts)


def _landmark_consistency_fallback(image_np: np.ndarray) -> float:
    """Fallback when the LBF model isn't available.

    Uses OpenCV eye cascades to measure basic bilateral symmetry
    of detected eye positions relative to the face centre.
    """
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml"
    )
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 5, minSize=(20, 20))

    if len(eyes) < 2:
        return 0.0

    # Sort eyes by area descending, take the two largest
    eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
    centres = [(ex + ew // 2, ey + eh // 2) for ex, ey, ew, eh in eyes]

    # Symmetry check: both eyes should be roughly equidistant from face centre
    face_cx = w / 2
    left_dx = abs(centres[0][0] - face_cx)
    right_dx = abs(centres[1][0] - face_cx)
    symmetry_ratio = min(left_dx, right_dx) / (max(left_dx, right_dx) + 1e-8)

    # Eye size ratio — should be close to 1.0 for real faces
    areas = [eyes[0][2] * eyes[0][3], eyes[1][2] * eyes[1][3]]
    size_ratio = min(areas) / (max(areas) + 1e-8)

    # Y-alignment — eyes should be at roughly the same height
    y_diff = abs(centres[0][1] - centres[1][1])
    y_score = min(y_diff / (h * 0.1 + 1e-8), 1.0)

    # Combine: lower ratios and higher y-offset → more likely fake
    inconsistency = (
        0.4 * (1.0 - symmetry_ratio)
        + 0.3 * (1.0 - size_ratio)
        + 0.3 * y_score
    )
    return float(np.clip(inconsistency, 0.0, 1.0))


def _analyse_landmark_geometry(pts: np.ndarray) -> float:
    """Analyse 68-point facial landmark geometry for GAN artifacts.

    Checks:
    1. Eye symmetry (left eye width vs right eye width)
    2. Nose-to-chin vs forehead-to-nose ratio
    3. Jawline smoothness (deviation from expected contour)
    4. Mouth-to-nose alignment
    """
    scores = []

    # --- Eye width symmetry (pts 36-41 = left eye, 42-47 = right eye) ---
    left_eye_w = np.linalg.norm(pts[36] - pts[39])
    right_eye_w = np.linalg.norm(pts[42] - pts[45])
    eye_ratio = min(left_eye_w, right_eye_w) / (max(left_eye_w, right_eye_w) + 1e-8)
    scores.append(1.0 - eye_ratio)  # 0 = perfect, 1 = very asymmetric

    # --- Eye vertical symmetry ---
    left_eye_cy = pts[36:42, 1].mean()
    right_eye_cy = pts[42:48, 1].mean()
    eye_y_diff = abs(left_eye_cy - right_eye_cy)
    face_height = np.linalg.norm(pts[8] - pts[27])  # chin to nose bridge
    scores.append(min(eye_y_diff / (face_height * 0.05 + 1e-8), 1.0))

    # --- Nose-mouth alignment (nose tip should be centred over mouth) ---
    nose_tip_x = pts[30][0]
    mouth_cx = (pts[48][0] + pts[54][0]) / 2
    mouth_w = np.linalg.norm(pts[48] - pts[54])
    alignment_off = abs(nose_tip_x - mouth_cx) / (mouth_w * 0.5 + 1e-8)
    scores.append(min(alignment_off, 1.0))

    # --- Jawline smoothness (pts 0-16) ---
    jawline = pts[0:17]
    diffs = np.diff(jawline, axis=0)
    angles = np.arctan2(diffs[:, 1], diffs[:, 0])
    angle_changes = np.abs(np.diff(angles))
    jaw_roughness = angle_changes.std()
    scores.append(min(jaw_roughness / 0.3, 1.0))  # normalized

    # Weighted combination
    weights = [0.30, 0.25, 0.20, 0.25]
    inconsistency = sum(w * s for w, s in zip(weights, scores))
    return float(np.clip(inconsistency, 0.0, 1.0))


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
    """Multi-band frequency analysis via 2-D FFT.

    GAN-generated faces leave spectral fingerprints across multiple
    frequency bands.  We analyse three bands (mid, high, ultra-high)
    and also measure spectral smoothness — GANs produce unnaturally
    uniform magnitude spectra compared to real camera images which
    have natural roll-off.

    Returns 0.0 (natural spectrum) to 1.0 (GAN-like spectrum).
    """
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # 2-D DFT, shift DC to centre
    dft = np.fft.fft2(gray)
    shifted = np.fft.fftshift(dft)
    magnitude = np.log1p(np.abs(shifted))  # log-magnitude for better dynamic range

    rows, cols = gray.shape
    cy, cx = rows // 2, cols // 2
    max_radius = min(cy, cx)

    y_coords, x_coords = np.ogrid[:rows, :cols]
    dist = np.sqrt((y_coords - cy) ** 2 + (x_coords - cx) ** 2)

    total_energy = magnitude.sum() + 1e-8

    # --- Multi-band energy ratios ---
    # Band 1: mid frequencies (15-30% of Nyquist) — structural patterns
    mid_mask = (dist > 0.15 * max_radius) & (dist <= 0.30 * max_radius)
    mid_energy = magnitude[mid_mask].sum() / total_energy

    # Band 2: high frequencies (30-60% of Nyquist) — texture/detail
    high_mask = (dist > 0.30 * max_radius) & (dist <= 0.60 * max_radius)
    high_energy = magnitude[high_mask].sum() / total_energy

    # Band 3: ultra-high frequencies (>60% of Nyquist) — noise/artifacts
    ultra_mask = dist > 0.60 * max_radius
    ultra_energy = magnitude[ultra_mask].sum() / total_energy

    # --- Spectral smoothness (azimuthal variance) ---
    # Real photos have natural directional variation; GANs are smoother.
    # Compute radial profile variance in the high-freq band.
    high_ring = magnitude.copy()
    high_ring[~high_mask] = 0
    # Divide into angular sectors and measure variance
    n_sectors = 16
    sector_energies = []
    for i in range(n_sectors):
        angle_low = 2 * np.pi * i / n_sectors - np.pi
        angle_high = 2 * np.pi * (i + 1) / n_sectors - np.pi
        angles = np.arctan2(y_coords - cy, x_coords - cx)
        sector_mask = (angles >= angle_low) & (angles < angle_high) & high_mask
        sector_energies.append(magnitude[sector_mask].sum())

    sector_energies = np.array(sector_energies)
    sector_mean = sector_energies.mean() + 1e-8
    spectral_uniformity = 1.0 - (sector_energies.std() / sector_mean)
    spectral_uniformity = float(np.clip(spectral_uniformity, 0.0, 1.0))

    # --- Combine sub-scores ---
    # Higher mid/high energy ratio with low ultra → GAN pattern
    # Steeper sigmoid (gain=30) with lower centre (0.42) for more sensitivity
    band_ratio = (mid_energy + high_energy) / (ultra_energy + 1e-8)
    band_score = 1.0 / (1.0 + np.exp(-30 * (high_energy - 0.42)))

    # Ultra-high flatness — GANs often have more energy here than real photos
    ultra_score = 1.0 / (1.0 + np.exp(-25 * (ultra_energy - 0.15)))

    # Weighted combination of frequency signals
    freq_score = (
        0.40 * band_score
        + 0.30 * ultra_score
        + 0.30 * spectral_uniformity
    )
    return float(np.clip(freq_score, 0.0, 1.0))


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

    # --- face landmark consistency (only when face detected) -----------------
    landmark_score = 0.0
    if face_detected:
        landmark_score = _landmark_consistency_score(display_image)
        # Blend landmark signal into freq_score (boosts GAN faces)
        freq_score = 0.65 * freq_score + 0.35 * landmark_score

    # --- adaptive weighted ensemble ------------------------------------------
    if face_detected:
        pw, fw = PIXEL_WEIGHT_FACE, FREQ_WEIGHT_FACE   # 0.6 / 0.4
    else:
        pw, fw = PIXEL_WEIGHT_DEFAULT, FREQ_WEIGHT_DEFAULT  # 0.7 / 0.3

    score = pw * pixel_score + fw * freq_score
    score = float(np.clip(score, 0.0, 1.0))
    confidence = max(score, 1.0 - score)

    return {
        "score": round(score, 4),
        "heatmap_base64": heatmap_b64,
        "freq_score": round(freq_score, 4),
        "pixel_score": round(pixel_score, 4),
        "landmark_score": round(landmark_score, 4),
        "confidence": round(confidence, 4),
        "face_detected": face_detected,
    }
