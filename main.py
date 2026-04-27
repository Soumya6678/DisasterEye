# =============================================================================
# main.py — DisasterEye FAST DEMO VERSION (~much higher FPS)
# =============================================================================

import cv2
import sys
import time
import argparse

from config import (
    INPUT_SOURCE, CSI_PIPELINE, JETSON_MODE, DISPLAY_WINDOW,
    SAVE_OUTPUT, OUTPUT_VIDEO, OUTPUT_FPS,
    ALERT_GPIO_PIN,
)

from inference import BodyPartDetector
from preprocessing import preprocess_frame
from utils import (
    draw_detections,
    draw_fps_and_inference,
    draw_survivor_alert,
    draw_scanning_status,
    draw_survivor_clusters,
    filter_small_detections,
    group_body_parts,
    count_survivors,
    FPSCounter,
    AlertTracker,
    create_video_writer
)

from logger import setup_logger, log_survivor, log_performance


# --------------------------------------------------
# GPIO (optional Jetson)
# --------------------------------------------------

def setup_gpio():
    try:
        import Jetson.GPIO as GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(ALERT_GPIO_PIN, GPIO.OUT)
        return GPIO
    except:
        return None


def trigger_alert_gpio(GPIO, pin):
    if GPIO:
        GPIO.output(pin, GPIO.HIGH)
        time.sleep(0.2)
        GPIO.output(pin, GPIO.LOW)


# --------------------------------------------------
# Camera
# --------------------------------------------------

def open_camera(source):
    if JETSON_MODE and source == 0:
        return cv2.VideoCapture(CSI_PIPELINE, cv2.CAP_GSTREAMER)
    return cv2.VideoCapture(source)


# --------------------------------------------------
# Main
# --------------------------------------------------

def run(source=None):

    source = source if source is not None else INPUT_SOURCE

    logger = setup_logger()

    cap = open_camera(source)

    if not cap.isOpened():
        print("Cannot open source")
        sys.exit(1)

    # LOWER RESOLUTION FOR FPS BOOST
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detector = BodyPartDetector()

    writer = create_video_writer(
        OUTPUT_VIDEO,w,h,OUTPUT_FPS
    ) if SAVE_OUTPUT else None

    fps_counter = FPSCounter()
    alert_tracker = AlertTracker()

    GPIO = setup_gpio() if JETSON_MODE else None

    frame_id=0
    confirmed_count=0

    # store detections between skipped frames
    detections=[]
    inference_ms=0

    print("Press q to quit")

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frame_id +=1

        enhanced = preprocess_frame(frame)

        # ========================================
        # RUN YOLO ONLY EVERY 3RD FRAME
        # ========================================
        if frame_id % 3 == 0:
            detections, inference_ms = detector.detect(enhanced)
            detections = filter_small_detections(detections)

        # Survivor clustering
        clusters = group_body_parts(detections)
        survivor_count = count_survivors(clusters)

        # simple alert logic
        new_alert = alert_tracker.update(survivor_count)

        if new_alert:
            confirmed_count = survivor_count
            log_survivor(
                logger,
                frame_id,
                [d["class_name"] for d in detections],
                max((d["confidence"] for d in detections), default=0),
                False
            )
            trigger_alert_gpio(GPIO, ALERT_GPIO_PIN)

        # ========================================
        # Draw overlays
        # ========================================
        display_frame = enhanced.copy()

        fps_counter.tick()
        fps = fps_counter.fps()

        display_frame = draw_detections(
            display_frame,
            detections
        )

        display_frame = draw_survivor_clusters(
            display_frame,
            clusters
        )

        display_frame = draw_fps_and_inference(
            display_frame,
            fps,
            inference_ms
        )

        if confirmed_count>0:
            display_frame = draw_survivor_alert(
                display_frame,
                confirmed_count
            )

        part_names = list(set(
            d["class_name"] for d in detections
        ))

        display_frame = draw_scanning_status(
            display_frame,
            part_names
        )

        # save output
        if SAVE_OUTPUT and writer:
            writer.write(display_frame)

        # show
        if DISPLAY_WINDOW:
            cv2.imshow(
                "DisasterEye FAST Demo",
                display_frame
            )

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # PERFORMANCE LOG ONLY EVERY 100 FRAMES
        if frame_id % 100 == 0:
            log_performance(
                logger,
                fps,
                detector.avg_inference_ms()
            )

    print("Done")

    cap.release()

    if writer:
        writer.release()

    cv2.destroyAllWindows()


# --------------------------------------------------
# Entry
# --------------------------------------------------

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=str,
        default=None
    )

    args = parser.parse_args()

    source = (
        int(args.source)
        if args.source and args.source.isdigit()
        else args.source
    )

    run(source)
