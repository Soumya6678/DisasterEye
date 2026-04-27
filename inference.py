# =============================================================================
# inference.py — DisasterEye
# Idea 1: YOLOv8s body part detection
# Idea 4: YOLOv8-Pose skeleton validation (PC only)
# =============================================================================

import time
import cv2
import numpy as np
from ultralytics import YOLO
from config import (
    YOLO_MODEL_PATH, POSE_MODEL_PATH,
    CONFIDENCE_THRESH, IOU_THRESH, INPUT_SIZE,
    DEVICE, CLASS_NAMES, USE_TENSORRT, TENSORRT_ENGINE_PATH,
    POSE_ENABLED, POSE_CONFIDENCE_THRESH, MIN_KEYPOINTS_VISIBLE
)


# ---------------------------------------------------------------------------
# Body part detector (Idea 1)
# ---------------------------------------------------------------------------

class BodyPartDetector:
    """
    YOLOv8s fine-tuned on body part classes:
    head, hand, torso, leg

    Detects partial body regions visible through rubble/debris.
    Even a single visible hand or head can trigger further analysis.
    """

    def __init__(self):
        path = TENSORRT_ENGINE_PATH if USE_TENSORRT else YOLO_MODEL_PATH
        print(f"[inference] Loading body part model: {path}")
        self.model  = YOLO(path)
        self.device = DEVICE

        # Warm-up
        dummy = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
        self.model(dummy, verbose=False)
        print("[inference] Body part model ready.")

        self._frames   = 0
        self._total_ms = 0.0

    def detect(self, frame: np.ndarray) -> tuple:
        """
        Run body part detection on frame.

        Returns:
            detections   : list of dicts {class_id, class_name, confidence, bbox}
            inference_ms : time in milliseconds
        """
        t0 = time.perf_counter()

        results = self.model.predict(
            source  = frame,
            conf    = CONFIDENCE_THRESH,
            iou     = IOU_THRESH,
            imgsz   = INPUT_SIZE,
            device  = self.device,
            verbose = False,
        )

        inference_ms = (time.perf_counter() - t0) * 1000.0
        self._total_ms += inference_ms
        self._frames   += 1

        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detections.append({
                    "class_id"   : cls_id,
                    "class_name" : CLASS_NAMES.get(cls_id, f"class_{cls_id}"),
                    "confidence" : conf,
                    "bbox"       : (x1, y1, x2, y2),
                })
        print(detections)
        return detections, inference_ms

    def avg_inference_ms(self) -> float:
        return self._total_ms / self._frames if self._frames > 0 else 0.0


# ---------------------------------------------------------------------------
# Pose estimator (Idea 4) — PC only
# ---------------------------------------------------------------------------

class PoseValidator:
    """
    YOLOv8-Pose skeleton validation.
    Verifies that detected body parts form a valid anatomical skeleton.

    Logic:
        Rubble cannot have: head → shoulder → elbow → wrist chain
        If keypoints form a plausible human chain → high confidence survivor

    NOTE: Disabled on Jetson Nano (too slow). PC demo only.
    """

    # COCO keypoint indices
    KEYPOINT_NAMES = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]

    # Anatomical chain: head → torso → limbs
    SKELETON_CHAINS = [
        [0, 5, 7, 9],    # nose → left_shoulder → left_elbow → left_wrist
        [0, 6, 8, 10],   # nose → right_shoulder → right_elbow → right_wrist
        [5, 11, 13, 15], # left_shoulder → left_hip → left_knee → left_ankle
        [6, 12, 14, 16], # right_shoulder → right_hip → right_knee → right_ankle
    ]

    def __init__(self):
        if not POSE_ENABLED:
            self.model = None
            print("[inference] Pose estimation DISABLED (Jetson Nano mode)")
            return

        print(f"[inference] Loading pose model: {POSE_MODEL_PATH}")
        self.model = YOLO(POSE_MODEL_PATH)
        dummy = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
        self.model(dummy, verbose=False)
        print("[inference] Pose model ready.")

    def validate(self, frame: np.ndarray) -> tuple:
        """
        Run pose estimation and validate skeleton chains.

        Returns:
            skeleton_confirmed : bool — True if valid human skeleton found
            keypoint_count     : int  — number of visible keypoints
            annotated_frame    : frame with skeleton drawn
        """
        if not POSE_ENABLED or self.model is None:
            return False, 0, frame

        results = self.model.predict(
            source  = frame,
            conf    = POSE_CONFIDENCE_THRESH,
            verbose = False,
            device  = DEVICE,
        )

        skeleton_confirmed = False
        total_keypoints    = 0

        for result in results:
            if result.keypoints is None:
                continue

            kps = result.keypoints.xy[0].cpu().numpy()    # (17, 2)
            kp_conf = result.keypoints.conf               # confidence per keypoint

            if kp_conf is not None:
                kp_conf = kp_conf[0].cpu().numpy()
                visible = np.sum(kp_conf > POSE_CONFIDENCE_THRESH)
            else:
                visible = np.sum(np.any(kps > 0, axis=1))

            total_keypoints = max(total_keypoints, int(visible))

            # Check anatomical chain validity
            for chain in self.SKELETON_CHAINS:
                chain_valid = True
                for idx in chain:
                    if kp_conf is not None and kp_conf[idx] < POSE_CONFIDENCE_THRESH:
                        chain_valid = False
                        break
                    if kps[idx][0] == 0 and kps[idx][1] == 0:
                        chain_valid = False
                        break
                if chain_valid:
                    skeleton_confirmed = True
                    break

            # Draw skeleton on frame
            frame = self._draw_skeleton(frame, kps,
                                        kp_conf if kp_conf is not None else None)

        return skeleton_confirmed, total_keypoints, frame

    def _draw_skeleton(self, frame, kps, kp_conf):
        """Draw detected keypoints and skeleton connections."""
        connections = [
            (0,1),(0,2),(1,3),(2,4),        # face
            (5,6),(5,7),(7,9),(6,8),(8,10), # arms
            (5,11),(6,12),(11,12),          # torso
            (11,13),(13,15),(12,14),(14,16) # legs
        ]

        for i, (x, y) in enumerate(kps):
            if x == 0 and y == 0:
                continue
            if kp_conf is not None and kp_conf[i] < POSE_CONFIDENCE_THRESH:
                continue
            cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 255), -1)

        for i, j in connections:
            xi, yi = kps[i]
            xj, yj = kps[j]
            if xi == 0 or xj == 0:
                continue
            cv2.line(frame,
                     (int(xi), int(yi)),
                     (int(xj), int(yj)),
                     (255, 255, 0), 1)

        return frame
