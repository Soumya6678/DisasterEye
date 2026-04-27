# =============================================================================
# test.py — DisasterEye
# Run inference on a static image or video file
# Usage:
#   python test.py --source image.jpg
#   python test.py --source video.mp4
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
 
 
# ---------------------------------------------------------------------------
# Shared: parse results into detection dicts
# ---------------------------------------------------------------------------
 
def parse_results(results):
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
    return detections
 
 
def annotate(frame, detections):
    clusters  = group_body_parts(detections)
    survivors = count_survivors(clusters)
    frame = draw_detections(frame, detections)
    frame = draw_survivor_clusters(frame, clusters)
    if survivors > 0:
        frame = draw_survivor_alert(frame, survivors)
    parts = list(set(d["class_name"] for d in detections))
    frame = draw_scanning_status(frame, parts)
    return frame, survivors
 
 
# ---------------------------------------------------------------------------
# Image mode
# ---------------------------------------------------------------------------
 
def run_on_image(model, path: str):
    frame = cv2.imread(path)
    if frame is None:
        print(f"❌ Cannot read image: {path}")
        return
 
    enhanced   = preprocess_frame(frame)
    results    = model.predict(source=enhanced, conf=CONFIDENCE_THRESH,
                               iou=IOU_THRESH, imgsz=INPUT_SIZE, verbose=False)
    detections = filter_small_detections(parse_results(results))
    frame, survivors = annotate(enhanced, detections)
 
    print(f"\n{'='*50}")
    print(f"  Detections : {len(detections)}")
    print(f"  Survivors  : {survivors}")
    print(f"{'='*50}")
    for d in detections:
        print(f"  {d['class_name']:<8}  conf={d['confidence']:.2f}  bbox={d['bbox']}")
 
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, "result_" + os.path.basename(path))
    cv2.imwrite(out, frame)
    print(f"\n💾 Saved: {out}")
 
    cv2.imshow("DisasterEye — Image Test", frame)
    print("Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
 
 
# ---------------------------------------------------------------------------
# Video mode
# ---------------------------------------------------------------------------
 
def run_on_video(model, path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {path}")
        return
 
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"▶ Video: {path}  [{w}×{h}]")
    print("  Press 'q' to quit | 's' to save snapshot")
 
    fps_counter = FPSCounter()
    frame_id    = 0
 
    while True:
        ret, frame = cap.read()
        if not ret:
            break
 
        frame_id += 1
        enhanced   = preprocess_frame(frame)
        results    = model.predict(source=enhanced, conf=CONFIDENCE_THRESH,
                                   iou=IOU_THRESH, imgsz=INPUT_SIZE, verbose=False)
        detections = filter_small_detections(parse_results(results))
        display, survivors = annotate(enhanced, detections)
 
        fps_counter.tick()
        display = draw_fps_and_inference(display, fps_counter.fps(), 0)
 
        cv2.imshow("DisasterEye — Video Test", display)
        key = cv2.waitKey(1) & 0xFF
 
        if key == ord('q'):
            break
        if key == ord('s'):
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            snap = os.path.join(OUTPUT_DIR, f"snap_{frame_id}.jpg")
            cv2.imwrite(snap, display)
            print(f"📸 Snapshot: {snap}")
 
    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Done — {frame_id} frames processed.")
 
 
# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DisasterEye — Test on image/video")
    parser.add_argument("--source", type=str, required=True,
                        help="Path to image (.jpg/.png) or video (.mp4/.avi)")
    args = parser.parse_args()
 
    if not os.path.exists(args.source):
        print(f"❌ File not found: {args.source}")
        sys.exit(1)
 
    print(f"[test] Loading model: {YOLO_MODEL_PATH}")
    model = YOLO(YOLO_MODEL_PATH)
    print("[test] ✅ Model ready\n")
 
    ext = os.path.splitext(args.source)[-1].lower()
    if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
        run_on_image(model, args.source)
    elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
        run_on_video(model, args.source)
    else:
        print(f"❌ Unsupported format: {ext}")
 