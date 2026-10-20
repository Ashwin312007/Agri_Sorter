"""Live multi-apple vision for the Raspberry Pi 4.

Stage 1 uses a YOLO detector to find and track every apple in a frame.
Stage 2 runs the trained fresh/rotten classifier on the crop for each tracked apple.
The resulting vision track ID is the ID that the encoder/ultrasonic integration must
associate with the upstream ultrasonic measurement before sorting.
"""
import argparse
import csv
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
from ultralytics import YOLO

BAD_LABELS = {"rotten", "rot", "bruise", "spot", "mold", "mould", "damage", "damaged"}


def find_class_id(model, class_name):
    for class_id, name in model.names.items():
        if name.lower() == class_name.lower():
            return class_id
    raise ValueError(f"Detector does not contain the class '{class_name}'. Available: {model.names}")


def crop_box(frame, box, padding=8):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (int(value) for value in box)
    x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
    x2, y2 = min(width, x2 + padding), min(height, y2 + padding)
    return frame[y1:y2, x1:x2]


def classify_apple(model, crop):
    result = model.predict(crop, imgsz=224, verbose=False)[0]
    label = result.names[result.probs.top1].lower()
    return label, float(result.probs.top1conf)


@dataclass
class TrackState:
    labels: Counter = field(default_factory=Counter)
    confidences: list = field(default_factory=list)
    emitted: bool = False
    last_seen: float = 0.0


class TrackDecisions:
    """Stabilise classifications across several frames of the same apple."""

    def __init__(self, minimum_frames):
        self.minimum_frames = minimum_frames
        self.items = {}

    def observe(self, track_id, label, confidence):
        state = self.items.setdefault(track_id, TrackState())
        state.labels[label] += 1
        state.confidences.append(confidence)
        state.last_seen = time.monotonic()
        final_label, votes = state.labels.most_common(1)[0]
        mean_confidence = sum(state.confidences) / len(state.confidences)
        return final_label, mean_confidence, votes >= self.minimum_frames and not state.emitted

    def mark_emitted(self, track_id):
        self.items[track_id].emitted = True

    def purge_old(self, seconds=5):
        now = time.monotonic()
        for track_id in list(self.items):
            if now - self.items[track_id].last_seen > seconds:
                del self.items[track_id]


def open_camera(args):
    if args.camera_backend in ("auto", "picamera2"):
        try:
            from picamera2 import Picamera2
            picam = Picamera2()
            config = picam.create_preview_configuration(main={"size": (1280, 720), "format": "RGB888"})
            picam.configure(config)
            picam.start()
            print("Using Raspberry Pi Camera Module via Picamera2")
            return picam, None
        except ImportError:
            if args.camera_backend == "picamera2":
                raise RuntimeError("Picamera2 is not installed. Use a USB camera or install Picamera2.")
        except Exception as error:
            if args.camera_backend == "picamera2":
                raise RuntimeError(f"Could not start Pi camera: {error}") from error
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError("Could not open USB camera")
    print("Using USB webcam via OpenCV")
    return None, camera


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector-model", required=True, help="YOLO detection weights with an 'apple' class.")
    parser.add_argument("--classifier-model", required=True, help="Fresh/rotten apple classifier weights.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--camera-backend", choices=("auto", "opencv", "picamera2"), default="auto")
    parser.add_argument("--minimum-frames", type=int, default=3, help="Matching classifications required before emitting a decision.")
    parser.add_argument("--log", default="runtime/vision_results.csv")
    args = parser.parse_args()

    detector = YOLO(args.detector_model)
    classifier = YOLO(args.classifier_model)
    apple_class = find_class_id(detector, "apple")
    decisions = TrackDecisions(args.minimum_frames)
    picam, camera = open_camera(args)
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not log_path.exists()

    with open(log_path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("timestamp", "vision_track_id", "external_label", "confidence", "vision_decision"))
        if is_new:
            writer.writeheader()
        print("Running multi-apple vision. Press q to stop.")
        while True:
            if picam is not None:
                frame = cv2.cvtColor(picam.capture_array(), cv2.COLOR_RGB2BGR)
            else:
                ok, frame = camera.read()
                if not ok:
                    continue

            result = detector.track(frame, persist=True, classes=[apple_class], imgsz=416,
                                    conf=0.30, tracker="bytetrack.yaml", verbose=False)[0]
            if result.boxes is not None and result.boxes.id is not None:
                for box, track_id in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.id.int().cpu().tolist()):
                    crop = crop_box(frame, box)
                    if crop.size == 0:
                        continue
                    label, confidence = classify_apple(classifier, crop)
                    final_label, mean_confidence, ready = decisions.observe(track_id, label, confidence)
                    vision_decision = "SECOND_QUALITY" if final_label in BAD_LABELS else "SIZE_GRADING"
                    x1, y1, x2, y2 = (int(value) for value in box)
                    color = (0, 0, 255) if vision_decision == "SECOND_QUALITY" else (0, 200, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"#{track_id} {final_label} {mean_confidence:.0%}", (x1, max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)
                    if ready:
                        writer.writerow({"timestamp": time.time(), "vision_track_id": track_id,
                                         "external_label": final_label, "confidence": f"{mean_confidence:.4f}",
                                         "vision_decision": vision_decision})
                        handle.flush()
                        decisions.mark_emitted(track_id)
                        print(f"Vision track #{track_id}: {vision_decision} ({final_label}, {mean_confidence:.0%})")
            decisions.purge_old()
            cv2.imshow("Fruit sorter: multi-apple vision", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    if picam is not None:
        picam.stop()
    if camera is not None:
        camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
