# =============================================================================
# download_models.py — DisasterEye
# Downloads the required Ultralytics model files into models/
# Run ONCE before using the system:
#   python download_models.py
# =============================================================================

import os
from ultralytics import YOLO

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def download(model_name: str):
    dest = os.path.join(MODEL_DIR, model_name)
    if os.path.exists(dest):
        print(f"✅ Already exists: {dest}")
        return
    print(f"⬇ Downloading {model_name} ...")
    model = YOLO(model_name)          # downloads to CWD
    if os.path.exists(model_name):
        os.rename(model_name, dest)
        print(f"✅ Saved to: {dest}")
    else:
        print(f"⚠️  {model_name} already in cache — copy manually to {MODEL_DIR}")


if __name__ == "__main__":
    print("=" * 50)
    print("DisasterEye — Model Downloader")
    print("=" * 50)

    # Pose model needed for Idea 4 (POSE_ENABLED=True)
    download("yolov8n-pose.pt")

    # Base training model (needed for training.py)
    download("yolov8s.pt")

    print("\n📋 Next step:")
    print(f"   Copy your trained best.pt → {MODEL_DIR}/best.pt")
    print("   (You already have it from the upload)")