# =============================================================================
# config.py — DisasterEye: Multi-Modal Survivor Detection System
# =============================================================================
 
import os
 
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
LOG_DIR    = os.path.join(BASE_DIR, "logs")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
 
# ---------------------------------------------------------------------------
# MODEL PATHS
# ---------------------------------------------------------------------------
YOLO_MODEL_PATH      = os.path.join(MODEL_DIR, "best.pt")         # VictimDet trained model
POSE_MODEL_PATH      = os.path.join(MODEL_DIR, "yolov8n-pose.pt") # skeleton validator
TENSORRT_ENGINE_PATH = os.path.join(MODEL_DIR, "best.engine")     # Jetson deployment
 
# ---------------------------------------------------------------------------
# INPUT SOURCE
# ---------------------------------------------------------------------------
# 0           = USB webcam
# "video.mp4" = test video file
# "nvarguscamerasrc ..." = CSI camera on Jetson Nano (set JETSON_MODE=True)
INPUT_SOURCE = 0
 
# CSI camera GStreamer pipeline for Jetson Nano
CSI_PIPELINE = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! "
    "nvvidconv flip-method=0 ! "
    "video/x-raw, width=640, height=480, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! appsink"
)
 
# ---------------------------------------------------------------------------
# MODEL PARAMETERS
# ---------------------------------------------------------------------------
INPUT_SIZE        = 450
CONFIDENCE_THRESH = 0.50   # lowered from 0.60 — VictimDet needs slightly lower threshold
IOU_THRESH        = 0.50
DEVICE            = "cpu"  # "0" for GPU on Jetson
 
# VictimDet class names (from trained model: model.names = {0: '0', 1: 'person'})
CLASS_NAMES = {
    0: "non-person",   # class '0' from dataset
    1: "person",       # survivor/victim
}
 
# BGR colours per class
CLASS_COLORS = {
    0: (100, 100, 100),  # gray  — non-person
    1: (0,   0,   255),  # red   — person (survivor)
}
 
# ---------------------------------------------------------------------------
# SURVIVOR DETECTION LOGIC
# ---------------------------------------------------------------------------
# For VictimDet: a single 'person' detection = potential survivor
# MIN_PARTS_FOR_SURVIVOR = 1 because we detect whole person, not body parts
MIN_PARTS_FOR_SURVIVOR  = 1
 
# Max pixel distance between detections to be considered "same person"
PROXIMITY_THRESHOLD_PX  = 200
 
# Consecutive frames a survivor must appear before confirmed alert
ALERT_FRAME_THRESHOLD   = 3
 
# Minimum bbox area to count as valid detection
MIN_BBOX_AREA = 500
 
# ---------------------------------------------------------------------------
# OPTICAL FLOW (Idea 2)
# ---------------------------------------------------------------------------
OPTICAL_FLOW_ENABLED     = True
MOTION_THRESHOLD         = 1.5   # mean magnitude to flag motion
FLOW_SCALE_FACTOR        = 0.5   # resize for faster flow computation
 
# ---------------------------------------------------------------------------
# GRADCAM (Idea 3 — simulated thermal)
# ---------------------------------------------------------------------------
GRADCAM_ENABLED          = True
GRADCAM_ALPHA            = 0.5   # overlay transparency
 
# ---------------------------------------------------------------------------
# POSE ESTIMATION (Idea 4 — PC only)
# ---------------------------------------------------------------------------
POSE_ENABLED             = False  # set True on PC demo, False on Jetson Nano
POSE_CONFIDENCE_THRESH   = 0.5
MIN_KEYPOINTS_VISIBLE    = 4      # minimum joints to confirm human skeleton
 
# ---------------------------------------------------------------------------
# OUTPUT / DISPLAY
# ---------------------------------------------------------------------------
DISPLAY_WINDOW  = True
SAVE_OUTPUT     = True
OUTPUT_FPS      = 20
OUTPUT_VIDEO    = os.path.join(OUTPUT_DIR, "disaster_eye_output.mp4")
LOG_FILE_PATH   = os.path.join(LOG_DIR,    "detections.log")
 
# ---------------------------------------------------------------------------
# JETSON NANO FLAGS
# ---------------------------------------------------------------------------
USE_TENSORRT   = False
JETSON_MODE    = False
ALERT_GPIO_PIN = 18    # BCM pin — red LED for survivor alert
SAFE_GPIO_PIN  = 23    # BCM pin — green LED for scanning
 
# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------
DATASET_YAML    = os.path.join(BASE_DIR, "data", "disaster.yaml")
EPOCHS          = 100
BATCH_SIZE      = 8
PRETRAINED_BASE = "yolov8m.pt"
TRAIN_IMGSZ     = 960
PROJECT_NAME    = "VictimDet"
EXPERIMENT_NAME = "yolov8m_victimdet"
