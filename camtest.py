# =============================================================================
# camtest.py — DisasterEye
# Live webcam inference — quick test on PC (VS Code)
# Usage:
#   python camtest.py            → default webcam (index 0)
#   python camtest.py --cam 1    → second camera
# =============================================================================

import argparse
import cv2
import os
import sys
from ultralytics import YOLO

from config import (
    YOLO_MODEL_PATH, CONFIDENCE_THRESH, IOU_THRESH,
    INPUT_SIZE, CLASS_NAMES, OUTPUT_DIR
)
from utils import (
    draw_detections, draw_fps_and_inference,
    draw_survivor_alert, draw_scanning_status,
    draw_survivor_clusters, filter_small_detections,
    group_body_parts, count_survivors, FPSCounter
)
from preprocessing import preprocess_frame


def run_camera(cam_index: int = 0):
    print(f"[camtest] Loading model: {YOLO_MODEL_PATH}")
    model = YOLO(YOLO_MODEL_PATH)
    print("[camtest] ✅ Model ready")

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"❌ Cannot open camera index {cam_index}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[camtest] Camera {cam_index} opened: {w}×{h}")
    print("[camtest] Press 'q' to quit | 's' to save snapshot\n")

    fps_counter = FPSCounter()
    frame_id    = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Frame read failed.")
            break

        frame_id += 1
        enhanced = preprocess_frame(frame)

        results = model.predict(
            source  = enhanced,
            conf    = CONFIDENCE_THRESH,
            iou     = IOU_THRESH,
            imgsz   = INPUT_SIZE,
            verbose = False
        )

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

        detections = filter_small_detections(detections)
        clusters   = group_body_parts(detections)
        survivors  = count_survivors(clusters)

        fps_counter.tick()
        fps = fps_counter.fps()

        display = draw_detections(enhanced, detections)
        display = draw_survivor_clusters(display, clusters)
        display = draw_fps_and_inference(display, fps, 0)
        if survivors > 0:
            display = draw_survivor_alert(display, survivors)
        parts   = list(set(d["class_name"] for d in detections))
        display = draw_scanning_status(display, parts)

        cv2.imshow("DisasterEye — Camera Test", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("[camtest] Quit.")
            break
        if key == ord('s'):
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            snap = os.path.join(OUTPUT_DIR, f"cam_snap_{frame_id}.jpg")
            cv2.imwrite(snap, display)
            print(f"📸 Snapshot saved: {snap}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DisasterEye — Camera Test")
    parser.add_argument("--cam", type=int, default=0,
                        help="Camera index (default: 0)")
    args = parser.parse_args()
    run_camera(args.cam)