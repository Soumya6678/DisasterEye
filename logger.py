# =============================================================================
# logger.py — DisasterEye
# =============================================================================

import logging, os
from datetime import datetime
from config import LOG_FILE_PATH, LOG_DIR


def setup_logger(name="DisasterEye"):
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh = logging.FileHandler(LOG_FILE_PATH)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info("=" * 60)
    logger.info(f"DisasterEye session started: {datetime.now()}")
    logger.info("=" * 60)
    return logger


def log_survivor(logger, frame_id, parts_detected, confidence, motion):
    logger.warning(
        f"*** SURVIVOR DETECTED *** Frame {frame_id} | "
        f"Parts: {parts_detected} | Conf: {confidence:.2f} | "
        f"Motion: {'YES' if motion else 'NO'}"
    )


def log_detection(logger, frame_id, detections):
    for d in detections:
        logger.info(
            f"Frame {frame_id:05d} | {d['class_name']:<6} | "
            f"Conf: {d['confidence']:.2f} | BBox: {d['bbox']}"
        )


def log_performance(logger, fps, inf_ms):
    logger.info(f"FPS: {fps:.1f} | Inference: {inf_ms:.1f} ms")


def log_system_info(logger, device, model, source):
    logger.info(f"Device: {device} | Model: {model} | Source: {source}")
