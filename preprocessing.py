# =============================================================================
# preprocessing.py — DisasterEye
# Input preparation + Optical Flow (Idea 2) + GradCAM heatmap (Idea 3)
# =============================================================================

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from config import (
    INPUT_SIZE, MOTION_THRESHOLD, FLOW_SCALE_FACTOR,
    GRADCAM_ALPHA, OPTICAL_FLOW_ENABLED
)


# ---------------------------------------------------------------------------
# Frame preprocessing
# ---------------------------------------------------------------------------

def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Enhance frame for low-light / dusty disaster conditions.
    Applies CLAHE contrast enhancement.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def resize_frame(frame: np.ndarray, width: int = 640) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = width / w
    return cv2.resize(frame, (width, int(h * scale)))


# ---------------------------------------------------------------------------
# IDEA 2 — Optical Flow Motion Detection
# ---------------------------------------------------------------------------

class OpticalFlowDetector:
    """
    Detects micro-motion in regions using Lucas-Kanade sparse optical flow.
    Rubble is static. Survivors show breathing/movement → non-zero flow.
    """

    def __init__(self):
        self.prev_gray   = None
        self.lk_params   = dict(
            winSize  = (15, 15),
            maxLevel = 2,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        self.feature_params = dict(
            maxCorners   = 100,
            qualityLevel = 0.3,
            minDistance  = 7,
            blockSize    = 7
        )

    def detect_motion(self, frame: np.ndarray, bbox: tuple) -> tuple[bool, float]:
        """
        Check if the region inside bbox has significant motion.

        Args:
            frame : current BGR frame
            bbox  : (x1, y1, x2, y2)

        Returns:
            (motion_detected: bool, motion_score: float)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x1, y1, x2, y2 = bbox

        # Crop region of interest
        roi_curr = gray[y1:y2, x1:x2]
        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            return False, 0.0

        roi_prev = self.prev_gray[y1:y2, x1:x2]
        if roi_curr.size == 0 or roi_prev.size == 0:
            self.prev_gray = gray.copy()
            return False, 0.0

        # Resize for speed
        scale = FLOW_SCALE_FACTOR
        small_prev = cv2.resize(roi_prev, None, fx=scale, fy=scale)
        small_curr = cv2.resize(roi_curr, None, fx=scale, fy=scale)

        # Dense optical flow (Farneback — more robust than sparse for small regions)
        flow = cv2.calcOpticalFlowFarneback(
            small_prev, small_curr, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion_score  = float(np.mean(magnitude))
        motion_detected = motion_score > MOTION_THRESHOLD

        self.prev_gray = gray.copy()
        return motion_detected, motion_score

    def draw_flow_overlay(self, frame: np.ndarray, bbox: tuple,
                          motion: bool, score: float) -> np.ndarray:
        """Draw motion indicator on the frame next to a bounding box."""
        x1, y1, x2, y2 = bbox
        color = (0, 255, 0) if motion else (100, 100, 100)
        label = f"MOTION:{score:.2f}" if motion else f"static:{score:.2f}"
        cv2.putText(frame, label, (x1, y2 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        return frame

    def reset(self):
        self.prev_gray = None


# ---------------------------------------------------------------------------
# IDEA 3 — GradCAM Heatmap (simulated thermal overlay)
# ---------------------------------------------------------------------------

class GradCAMGenerator:
    """
    Generates GradCAM activation heatmap from YOLOv8 backbone.
    Highlights regions the model "focuses on" — acts as pseudo-thermal overlay.
    Works with CSI camera (no actual thermal sensor needed).
    """

    def __init__(self, model):
        self.model      = model
        self.gradients  = None
        self.activations= None
        self._hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        """Hook into the last Conv layer of YOLOv8 backbone."""
        try:
            # YOLOv8 backbone last layer
            target_layer = self.model.model.model[-2]

            def forward_hook(module, input, output):
                self.activations = output.detach()

            def backward_hook(module, grad_in, grad_out):
                self.gradients = grad_out[0].detach()

            self._hook_handles.append(
                target_layer.register_forward_hook(forward_hook)
            )
            self._hook_handles.append(
                target_layer.register_full_backward_hook(backward_hook)
            )
        except Exception:
            pass  # GradCAM hooks not critical — fail silently

    def generate(self, frame: np.ndarray) -> np.ndarray:
        """
        Generate GradCAM heatmap for a given frame.

        Args:
            frame : BGR numpy array

        Returns:
            heatmap : BGR heatmap same size as frame, or None if unavailable
        """
        try:
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            tensor = tensor.unsqueeze(0)
            tensor.requires_grad_(True)

            output = self.model.model(tensor)

            # Use max class score as target for backprop
            if isinstance(output, (list, tuple)):
                score = output[0].max()
            else:
                score = output.max()

            self.model.model.zero_grad()
            score.backward(retain_graph=True)

            if self.gradients is None or self.activations is None:
                return None

            # Pool gradients across channels
            pooled_grads = torch.mean(self.gradients, dim=[0, 2, 3])
            activations  = self.activations[0]

            for i, w in enumerate(pooled_grads):
                activations[i] *= w

            heatmap = activations.mean(dim=0).numpy()
            heatmap = np.maximum(heatmap, 0)
            if heatmap.max() > 0:
                heatmap /= heatmap.max()

            # Resize to frame size
            h, w = frame.shape[:2]
            heatmap_resized = cv2.resize(heatmap, (w, h))
            heatmap_uint8   = np.uint8(255 * heatmap_resized)
            heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

            return heatmap_colored

        except Exception:
            return None

    def overlay(self, frame: np.ndarray, heatmap: np.ndarray,
                alpha: float = GRADCAM_ALPHA) -> np.ndarray:
        """Blend heatmap onto frame."""
        if heatmap is None:
            return frame
        return cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0)

    def remove_hooks(self):
        for h in self._hook_handles:
            h.remove()
