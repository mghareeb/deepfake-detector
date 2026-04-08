from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification, pipeline

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HF_MODEL_ID = os.environ.get(
    "HF_DEEPFAKE_MODEL", "buildborderless/CommunityForensics-DeepfakeDet-ViT"
)
FACE_MARGIN = 0.3

# Default ensemble weights (no face detected — full image)
PIXEL_WEIGHT_DEFAULT = 0.5
FREQ_WEIGHT_DEFAULT = 0.5

# Face-specific ensemble weights — pretrained deepfake model gets more trust
PIXEL_WEIGHT_FACE = 0.6
FREQ_WEIGHT_FACE = 0.4

# Calibration: when the deepfake model is very confident, trust it more
MODEL_CONFIDENCE_THRESHOLD = 0.8
CALIBRATION_BIAS = 0.10

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
    1.0 (highly inconsistent -> likely fake).
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

    # Combine: lower ratios and higher y-offset -> more likely fake
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


def _skin_texture_score(image_np: np.ndarray) -> float:
    """Detect unnaturally uniform skin texture typical of GAN output.

    Real skin has natural local variance from pores, fine hairs, moles,
    and micro-wrinkles.  GANs (especially StyleGAN2) produce skin regions
    that are locally *too smooth*.

    Returns 0.0 (natural texture) to 1.0 (GAN-like uniformity).
    """
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY).astype(np.float64)

    # Mild blur to suppress JPEG blocking artifacts (they add fake variance)
    gray = cv2.GaussianBlur(gray, (3, 3), 0.8)

    # Local mean and local variance via box filter
    ksize = 7
    local_mean = cv2.blur(gray, (ksize, ksize))
    local_sq_mean = cv2.blur(gray ** 2, (ksize, ksize))
    local_var = np.maximum(local_sq_mean - local_mean ** 2, 0.0)

    # Extract central 60% of the image (face interior, avoiding edges)
    h, w = local_var.shape
    y1, y2 = int(h * 0.20), int(h * 0.80)
    x1, x2 = int(w * 0.20), int(w * 0.80)
    centre_var = local_var[y1:y2, x1:x2]

    if centre_var.size == 0:
        return 0.0

    mean_var = centre_var.mean()
    std_var = centre_var.std()

    # --- Signal 1: overall low local variance (smooth skin) ---
    smoothness = 1.0 / (1.0 + np.exp(0.12 * (mean_var - 50)))

    # --- Signal 2: uniform variance across the face ---
    cov = std_var / (mean_var + 1e-8)
    uniformity = 1.0 / (1.0 + np.exp(8 * (cov - 0.9)))

    # --- Signal 3: lack of high-frequency micro-texture ---
    laplacian = cv2.Laplacian(gray[y1:y2, x1:x2], cv2.CV_64F)
    lap_std = laplacian.std()
    micro_texture = 1.0 / (1.0 + np.exp(0.5 * (lap_std - 8)))

    score = 0.35 * smoothness + 0.35 * uniformity + 0.30 * micro_texture
    return float(np.clip(score, 0.0, 1.0))


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


# ---------------------------------------------------------------------------
# Grad-CAM for transformer / CNN models from HuggingFace
# ---------------------------------------------------------------------------

def _find_target_layer(model):
    """Auto-discover the last attention/conv layer for Grad-CAM.

    Supports ViT (layernorm after last encoder block) and CNN-based
    models (last Conv2d layer).
    """
    # ViT-style: look for encoder blocks inside a vision backbone
    for attr in ("vit", "deit", "swin", "beit", "vision_model", "siglip"):
        backbone = getattr(model, attr, None)
        if backbone is not None:
            encoder = getattr(backbone, "encoder", None)
            if encoder is not None:
                # Different models use "layer" or "layers"
                layer_list = getattr(encoder, "layer", None) or getattr(encoder, "layers", None)
                if layer_list is not None and len(layer_list) > 0:
                    last_block = layer_list[-1]
                    # Prefer layernorm after attention (captures full attention output)
                    for ln_name in ("layernorm_after", "layer_norm2", "norm2", "ln_2"):
                        ln = getattr(last_block, ln_name, None)
                        if ln is not None:
                            return ln
                    # Fallback: the block itself
                    return last_block

    # CNN-style: find the last Conv2d
    last_conv = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    if last_conv is not None:
        return last_conv

    raise RuntimeError("Could not find a suitable Grad-CAM target layer")


class _GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handles = [
            target_layer.register_forward_hook(self._fwd),
            target_layer.register_full_backward_hook(self._bwd),
        ]

    def _fwd(self, _module, _input, output):
        # output may be a tensor or tuple
        out = output[0] if isinstance(output, tuple) else output
        self.activations = out.detach()

    def _bwd(self, _module, _grad_in, grad_out):
        out = grad_out[0] if isinstance(grad_out, tuple) else grad_out
        self.gradients = out.detach()

    def generate(
        self,
        pixel_values: torch.Tensor,
        original_image: np.ndarray,
        target_class: int = 1,
    ) -> str:
        """Return a base64-encoded PNG of the Grad-CAM heatmap overlaid on the image."""
        try:
            return self._generate_inner(pixel_values, original_image, target_class)
        except Exception as exc:
            print(f"[model] Grad-CAM failed ({exc}), returning blank heatmap", flush=True)
            return _blank_heatmap(original_image)

    def _generate_inner(
        self,
        pixel_values: torch.Tensor,
        original_image: np.ndarray,
        target_class: int = 1,
    ) -> str:
        self.model.eval()
        inp = pixel_values.clone().requires_grad_(True)

        with torch.enable_grad():
            outputs = self.model(pixel_values=inp)
            logits = outputs.logits
            self.model.zero_grad()
            logits[0, target_class].backward()

        if self.gradients is None or self.activations is None:
            return _blank_heatmap(original_image)

        acts = self.activations
        grads = self.gradients

        # ViT activations are (B, num_patches(+1), hidden_dim)
        if acts.dim() == 3:
            seq_len = acts.shape[1]
            # Check if there's a CLS token (seq_len is not a perfect square)
            grid_size = int(seq_len ** 0.5)
            if grid_size * grid_size != seq_len:
                # Has CLS token — remove it
                acts = acts[:, 1:, :]
                grads = grads[:, 1:, :]
                seq_len = acts.shape[1]
                grid_size = int(seq_len ** 0.5)

            weights = grads.mean(dim=2, keepdim=True)  # (B, seq_len, 1)
            cam = torch.relu((weights * acts).sum(dim=2))  # (B, seq_len)
            cam = cam.view(1, grid_size, grid_size)
        elif acts.dim() == 4:
            # Standard CNN: (B, C, H, W)
            weights = grads.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * acts).sum(dim=1, keepdim=True))
            cam = cam.squeeze(1)
        else:
            return _blank_heatmap(original_image)

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


def _blank_heatmap(image_np: np.ndarray) -> str:
    """Return a base64 PNG of the original image (no heatmap) as fallback."""
    _, buf = cv2.imencode(".png", cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf).decode("utf-8")


# ---------------------------------------------------------------------------
# Frequency analysis (numpy FFT)
# ---------------------------------------------------------------------------

def _frequency_score(image_np: np.ndarray) -> float:
    """Multi-band frequency analysis tuned for GAN artifact detection.

    StyleGAN / StyleGAN2 / ProGAN images leave characteristic spectral
    fingerprints:
      1. Periodic peaks from upsampling convolutions
      2. Abnormally high energy in mid-high frequency bands
      3. Unnaturally smooth (isotropic) azimuthal energy distribution
      4. Steeper-than-natural spectral roll-off in specific bands

    Returns 0.0 (natural spectrum) to 1.0 (GAN-like spectrum).
    """
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # 2-D DFT, shift DC to centre
    dft = np.fft.fft2(gray)
    shifted = np.fft.fftshift(dft)
    magnitude = np.log1p(np.abs(shifted))

    rows, cols = gray.shape
    cy, cx = rows // 2, cols // 2
    max_radius = min(cy, cx)

    y_coords, x_coords = np.ogrid[:rows, :cols]
    dist = np.sqrt((y_coords - cy) ** 2 + (x_coords - cx) ** 2)

    total_energy = magnitude.sum() + 1e-8

    # --- Multi-band energy ratios ---
    mid_mask = (dist > 0.10 * max_radius) & (dist <= 0.25 * max_radius)
    mid_energy = magnitude[mid_mask].sum() / total_energy

    high_mask = (dist > 0.25 * max_radius) & (dist <= 0.50 * max_radius)
    high_energy = magnitude[high_mask].sum() / total_energy

    ultra_mask = dist > 0.50 * max_radius
    ultra_energy = magnitude[ultra_mask].sum() / total_energy

    # --- Spectral roll-off anomaly ---
    non_dc_mask = dist > 0.05 * max_radius
    non_dc_energy = magnitude[non_dc_mask].sum() + 1e-8
    rolloff_ratio = (magnitude[mid_mask].sum() + magnitude[high_mask].sum()) / non_dc_energy

    # --- Azimuthal (angular) uniformity ---
    angles = np.arctan2(y_coords - cy, x_coords - cx)
    n_sectors = 24
    sector_energies = np.zeros(n_sectors)
    analysis_mask = (dist > 0.15 * max_radius) & (dist <= 0.60 * max_radius)
    for i in range(n_sectors):
        a_lo = 2 * np.pi * i / n_sectors - np.pi
        a_hi = 2 * np.pi * (i + 1) / n_sectors - np.pi
        s_mask = (angles >= a_lo) & (angles < a_hi) & analysis_mask
        sector_energies[i] = magnitude[s_mask].sum()

    sector_mean = sector_energies.mean() + 1e-8
    coeff_of_var = sector_energies.std() / sector_mean
    spectral_uniformity = 1.0 / (1.0 + np.exp(40 * (coeff_of_var - 0.12)))

    # --- Periodic peak detector ---
    n_bins = max(50, max_radius // 2)
    radial_profile = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    dist_flat = dist.ravel()
    mag_flat = magnitude.ravel()
    bin_edges = np.linspace(0, max_radius, n_bins + 1)
    for b in range(n_bins):
        mask = (dist_flat >= bin_edges[b]) & (dist_flat < bin_edges[b + 1])
        if mask.any():
            radial_profile[b] = mag_flat[mask].mean()
            bin_counts[b] = mask.sum()

    if radial_profile.max() > 0:
        smooth = np.convolve(radial_profile, np.ones(5) / 5, mode="same")
        residual = radial_profile - smooth
        peak_strength = np.abs(residual).mean() / (smooth.mean() + 1e-8)
    else:
        peak_strength = 0.0
    periodic_score = 1.0 / (1.0 + np.exp(-50 * (peak_strength - 0.04)))

    # --- Combine sub-scores ---
    band_score = 1.0 / (1.0 + np.exp(-40 * (high_energy - 0.35)))
    ultra_score = 1.0 / (1.0 + np.exp(-35 * (ultra_energy - 0.10)))
    rolloff_score = 1.0 / (1.0 + np.exp(-30 * (rolloff_ratio - 0.55)))

    freq_score = (
        0.25 * band_score
        + 0.15 * ultra_score
        + 0.25 * spectral_uniformity
        + 0.15 * rolloff_score
        + 0.20 * periodic_score
    )
    return float(np.clip(freq_score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Model lifecycle
# ---------------------------------------------------------------------------

def load_model() -> dict:
    """Initialise the HuggingFace deepfake detection model + Grad-CAM.

    Downloads the model and processor from HF Hub on first run (cached
    automatically by transformers).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[model] Loading HuggingFace model: {HF_MODEL_ID}")
    processor = AutoImageProcessor.from_pretrained(HF_MODEL_ID, trust_remote_code=True)
    # Fix image size for models that expect specific dimensions (e.g. 384x384)
    if hasattr(processor, "size"):
        model_size = 384  # ViT-base patch16 = 384
        processor.size = {"height": model_size, "width": model_size}
        if hasattr(processor, "crop_size"):
            processor.crop_size = {"height": model_size, "width": model_size}
    model = AutoModelForImageClassification.from_pretrained(HF_MODEL_ID, trust_remote_code=True)
    model.eval()
    model.to(device)

    # Create a pipeline — override image_processor to use our fixed one
    pipe = pipeline(
        "image-classification",
        model=model,
        image_processor=processor,
        device=device,
    )

    # Discover the label that means "fake"
    label2id = getattr(model.config, "label2id", {})
    fake_idx = None
    for label_name in ("fake", "Fake", "FAKE", "deepfake", "Deepfake", "artificial", "ai", "1"):
        if label_name in label2id:
            fake_idx = label2id[label_name]
            break
    if fake_idx is None:
        # CommunityForensics model: LABEL_0=real, LABEL_1=fake
        fake_idx = 1
    print(f"[model] Labels: {model.config.id2label}, fake_idx={fake_idx}")

    target_layer = _find_target_layer(model)
    print(f"[model] Grad-CAM target layer: {target_layer.__class__.__name__}")
    gradcam = _GradCAM(model, target_layer)

    global _model_state
    _model_state = {
        "model": model,
        "processor": processor,
        "device": device,
        "gradcam": gradcam,
        "fake_idx": fake_idx,
        "pipe": pipe,
    }
    return _model_state


def cleanup():
    """Remove Grad-CAM hooks (call on shutdown)."""
    if _model_state is not None:
        _model_state["gradcam"].remove_hooks()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict(image_path: str) -> dict:
    """Run the full detection pipeline on an image file."""
    image = Image.open(image_path).convert("RGB")
    return _run_pipeline(np.array(image))


def predict_from_bytes(image_bytes: bytes) -> dict:
    """Same as ``predict`` but accepts raw file bytes (used by the API)."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return _run_pipeline(np.array(image))


def _run_pipeline(image_np: np.ndarray) -> dict:
    state = _model_state
    if state is None:
        raise RuntimeError("Model not loaded -- call load_model() first")

    model = state["model"]
    processor = state["processor"]
    device: torch.device = state["device"]
    gradcam: _GradCAM = state["gradcam"]
    fake_idx: int = state["fake_idx"]
    pipe = state["pipe"]

    # --- face detection & crop ------------------------------------------------
    cropped, face_detected = _detect_and_crop_face(image_np)

    # Resize for display / frequency analysis
    display_size = 384
    display_image = cv2.resize(cropped, (display_size, display_size))

    # --- pixel-domain score (HF deepfake model) -------------------------------
    # Feed the FULL image to the model (not the cropped face) because
    # deepfake detection models are trained on full images and lose
    # accuracy when given tightly-cropped face regions.
    pil_image = Image.fromarray(image_np)

    # Use pipeline for reliable label-based scoring
    pipe_results = pipe(pil_image)
    print(f"[model] RAW: {pipe_results}", flush=True)

    # Extract fake score from pipeline results
    pixel_score = 0.5  # default
    for item in pipe_results:
        label_lower = item["label"].lower()
        if any(kw in label_lower for kw in ("fake", "artificial", "deepfake", "ai", "label_1")):
            pixel_score = item["score"]
            break
    else:
        # No fake/artificial label found — use 1 - real/human score
        for item in pipe_results:
            label_lower = item["label"].lower()
            if any(kw in label_lower for kw in ("real", "human", "hum", "label_0")):
                pixel_score = 1.0 - item["score"]
                break
    print(f"[model] pixel_score(fake): {pixel_score:.4f}", flush=True)

    # Also run through the model directly for Grad-CAM
    inputs = processor(images=pil_image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    # --- Grad-CAM heatmap -----------------------------------------------------
    heatmap_b64 = gradcam.generate(pixel_values, display_image, target_class=fake_idx)

    # --- frequency-domain score -----------------------------------------------
    freq_score_raw = _frequency_score(display_image)

    # --- face-specific heuristics (only when face detected) -------------------
    landmark_score = 0.0
    skin_score = 0.0
    freq_score = freq_score_raw

    if face_detected:
        landmark_score = _landmark_consistency_score(display_image)
        skin_score = _skin_texture_score(display_image)

        # Blend all frequency-domain signals
        freq_score = (
            0.50 * freq_score_raw
            + 0.25 * skin_score
            + 0.25 * landmark_score
        )

    # --- adaptive weighted ensemble ------------------------------------------
    if face_detected:
        pw, fw = PIXEL_WEIGHT_FACE, FREQ_WEIGHT_FACE   # 0.6 / 0.4
    else:
        pw, fw = PIXEL_WEIGHT_DEFAULT, FREQ_WEIGHT_DEFAULT  # 0.5 / 0.5

    score = pw * pixel_score + fw * freq_score

    # --- calibration bias ----------------------------------------------------
    if pixel_score > MODEL_CONFIDENCE_THRESHOLD:
        score += CALIBRATION_BIAS

    score = float(np.clip(score, 0.0, 1.0))
    confidence = max(score, 1.0 - score)

    return {
        "score": round(score, 4),
        "heatmap_base64": heatmap_b64,
        "freq_score": round(freq_score, 4),
        "pixel_score": round(pixel_score, 4),
        "landmark_score": round(landmark_score, 4),
        "skin_score": round(skin_score, 4),
        "confidence": round(confidence, 4),
        "face_detected": face_detected,
    }
