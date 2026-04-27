# =============================================================================
# utils.py — DisasterEye
# Drawing, survivor proximity logic, alert tracking, FPS counter
# =============================================================================

import cv2
import time
import numpy as np
import os
from config import (
    CLASS_COLORS, CLASS_NAMES, MIN_BBOX_AREA,
    PROXIMITY_THRESHOLD_PX, MIN_PARTS_FOR_SURVIVOR,
    ALERT_FRAME_THRESHOLD, OUTPUT_DIR
)


# ---------------------------------------------------------------------------
# Bounding box drawing
# ---------------------------------------------------------------------------

def draw_detections(frame: np.ndarray, detections: list) -> np.ndarray:
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cls_id = det["class_id"]
        label  = f"{det['class_name']} {det['confidence']:.2f}"
        color  = CLASS_COLORS.get(cls_id, (255, 255, 255))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame


def draw_fps_and_inference(frame: np.ndarray, fps: float, inf_ms: float) -> np.ndarray:
    cv2.putText(frame, f"FPS: {fps:.1f}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.putText(frame, f"Inference: {inf_ms:.1f} ms",
                (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    return frame


def draw_survivor_alert(frame: np.ndarray, count: int) -> np.ndarray:
    """Draw red alert banner when survivor is confirmed."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 55), (0, 0, 200), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(
        frame,
        f"  *** SURVIVOR DETECTED — {count} PERSON(S) — ALERT TRIGGERED ***",
        (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2
    )
    return frame


def draw_scanning_status(frame: np.ndarray, parts: list) -> np.ndarray:
    """Show scanning status and detected parts in bottom bar."""
    h, w = frame.shape[:2]
    part_str = " | ".join(parts) if parts else "scanning..."
    cv2.putText(frame, f"Parts: {part_str}",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (200, 200, 200), 1)
    return frame


# ---------------------------------------------------------------------------
# IDEA 1 — Survivor proximity logic
# ---------------------------------------------------------------------------

def bbox_center(bbox: tuple) -> tuple:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def euclidean(p1: tuple, p2: tuple) -> float:
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) ** 0.5


def group_body_parts(detections: list,
                     proximity_px: int = PROXIMITY_THRESHOLD_PX) -> list:
    """
    Group nearby body part detections into potential survivor clusters.

    Logic:
        - Compute center of each detected body part bbox
        - If two or more body parts are within proximity_px of each other
          → they likely belong to the same person
        - Each cluster = one potential survivor

    Args:
        detections   : list of detection dicts
        proximity_px : max pixel distance to group parts together

    Returns:
        List of clusters, each cluster is a list of detection dicts.
    """
    if not detections:
        return []

    centers   = [bbox_center(d["bbox"]) for d in detections]
    visited   = [False] * len(detections)
    clusters  = []

    for i in range(len(detections)):
        if visited[i]:
            continue
        cluster = [detections[i]]
        visited[i] = True
        for j in range(i + 1, len(detections)):
            if not visited[j]:
                if euclidean(centers[i], centers[j]) < proximity_px:
                    cluster.append(detections[j])
                    visited[j] = True
        clusters.append(cluster)

    return clusters


def count_survivors(clusters: list,
                    min_parts: int = MIN_PARTS_FOR_SURVIVOR) -> int:
    """
    Count clusters that have enough unique body parts to be a survivor.

    Args:
        clusters  : output of group_body_parts()
        min_parts : minimum unique body part types needed

    Returns:
        Number of likely survivors detected.
    """
    count = 0
    for cluster in clusters:
        unique_parts = set(d["class_name"] for d in cluster)
        if len(unique_parts) >= min_parts:
            count += 1
    return count


def draw_survivor_clusters(frame: np.ndarray, clusters: list,
                            min_parts: int = MIN_PARTS_FOR_SURVIVOR) -> np.ndarray:
    """Draw a circle grouping each confirmed survivor cluster."""
    for cluster in clusters:
        unique_parts = set(d["class_name"] for d in cluster)
        if len(unique_parts) < min_parts:
            continue

        # Find bounding rect of the whole cluster
        all_x1 = min(d["bbox"][0] for d in cluster)
        all_y1 = min(d["bbox"][1] for d in cluster)
        all_x2 = max(d["bbox"][2] for d in cluster)
        all_y2 = max(d["bbox"][3] for d in cluster)

        cx = (all_x1 + all_x2) // 2
        cy = (all_y1 + all_y2) // 2
        radius = max((all_x2 - all_x1), (all_y2 - all_y1)) // 2 + 20

        cv2.circle(frame, (cx, cy), radius, (0, 0, 255), 2)
        cv2.putText(frame, "SURVIVOR?",
                    (cx - 40, cy - radius - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return frame


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_small_detections(detections: list,
                             min_area: int = MIN_BBOX_AREA) -> list:
    return [d for d in detections
            if (d["bbox"][2]-d["bbox"][0]) * (d["bbox"][3]-d["bbox"][1]) >= min_area]


# ---------------------------------------------------------------------------
# FPS counter
# ---------------------------------------------------------------------------

class FPSCounter:
    def __init__(self, window: int = 30):
        self.window = window
        self._ts    = []

    def tick(self):
        self._ts.append(time.time())
        if len(self._ts) > self.window:
            self._ts.pop(0)

    def fps(self) -> float:
        if len(self._ts) < 2:
            return 0.0
        elapsed = self._ts[-1] - self._ts[0]
        return (len(self._ts) - 1) / elapsed if elapsed > 0 else 0.0


# ---------------------------------------------------------------------------
# Alert tracker
# ---------------------------------------------------------------------------

class AlertTracker:
    def __init__(self, threshold: int = ALERT_FRAME_THRESHOLD):
        self.threshold = threshold
        self._count    = 0
        self._alerted  = False

    def update(self, survivor_count: int) -> bool:
        if survivor_count > 0:
            self._count += 1
            if self._count >= self.threshold and not self._alerted:
                self._alerted = True
                return True
        else:
            self._count   = 0
            self._alerted = False
        return False

    def reset(self):
        self._count   = 0
        self._alerted = False


# ---------------------------------------------------------------------------
# Video writer
# ---------------------------------------------------------------------------

def create_video_writer(path: str, w: int, h: int, fps: int = 20):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, (w, h))
