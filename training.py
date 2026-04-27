# =============================================================================
# training.py — DisasterEye
# Fine-tunes YOLOv8s on body part dataset for disaster survivor detection
# Run on Google Colab with T4 GPU — NOT on Jetson Nano
# =============================================================================

import os, shutil
from ultralytics import YOLO
from config import (
    DATASET_YAML, EPOCHS, BATCH_SIZE, PRETRAINED_BASE,
    TRAIN_IMGSZ, PROJECT_NAME, EXPERIMENT_NAME,
    YOLO_MODEL_PATH, MODEL_DIR
)


def describe_model():
    print("\n" + "="*60)
    print("DisasterEye — YOLOv8s Body Part Detector")
    print("="*60)
    print("  Classes    : head, hand, torso, leg")
    print("  Backbone   : CSPDarknet with C2f modules")
    print("  Neck       : PAN-FPN")
    print("  Head       : Decoupled anchor-free detection head")
    print("  Input      : 640×640 RGB")
    print("  Parameters : ~11.2M")
    print("  Framework  : Ultralytics (PyTorch)")
    print("="*60 + "\n")


def train():
    model = YOLO(PRETRAINED_BASE)
    describe_model()

    results = model.train(
        data         = DATASET_YAML,
        epochs       = EPOCHS,
        imgsz        = TRAIN_IMGSZ,
        batch        = BATCH_SIZE,
        device       = 0,
        optimizer    = "AdamW",
        lr0          = 0.001,
        lrf          = 0.01,
        momentum     = 0.937,
        weight_decay = 0.0005,
        warmup_epochs= 3,
        cos_lr       = True,
        patience     = 15,
        augment      = True,
        mosaic       = 1.0,
        mixup        = 0.1,
        hsv_h        = 0.015,
        hsv_s        = 0.7,
        hsv_v        = 0.4,
        fliplr       = 0.5,
        degrees      = 10.0,
        project      = PROJECT_NAME,
        name         = EXPERIMENT_NAME,
        save         = True,
        save_period  = 10,
        val          = True,
        plots        = True,
        verbose      = True,
    )

    best = os.path.join(PROJECT_NAME, EXPERIMENT_NAME, "weights", "best.pt")
    if os.path.exists(best):
        os.makedirs(MODEL_DIR, exist_ok=True)
        shutil.copy2(best, os.path.join(MODEL_DIR, "best.pt"))
        print(f"\n✅ Best weights saved to: {MODEL_DIR}/best.pt")

    return results


def validate(model_path=None):
    path    = model_path or YOLO_MODEL_PATH
    model   = YOLO(path)
    metrics = model.val(data=DATASET_YAML, imgsz=TRAIN_IMGSZ, device="cpu", plots=True)

    print("\nVALIDATION RESULTS")
    print(f"  mAP@50    : {metrics.box.map50:.4f}")
    print(f"  mAP@50-95 : {metrics.box.map:.4f}")
    print(f"  Precision : {metrics.box.mp:.4f}")
    print(f"  Recall    : {metrics.box.mr:.4f}")
    return metrics


def export_tensorrt(model_path=None):
    path  = model_path or YOLO_MODEL_PATH
    model = YOLO(path)
    model.export(format="engine", imgsz=TRAIN_IMGSZ, half=True, device=0, simplify=True)
    print("✅ TensorRT .engine file saved")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train","val","export"], default="train")
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    if args.mode == "train":   train()
    elif args.mode == "val":   validate(args.model)
    elif args.mode == "export":export_tensorrt(args.model)
