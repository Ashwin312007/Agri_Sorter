"""Main Sorter Application for Agri_Sorter.

Integrates:
- Camera detection / synthetic video conveyor ingestion (Station 0)
- Conveyor ByteTrack & Sticky Decision tracking (camera egress queue)
- HC-SR04 ultrasonic triggering (Station 1)
- Diverter flap servo actuation & fail-safe rejection (Station 2)
- Comprehensive simulation mode (--simulate) for end-to-end desktop verification.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import queue
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Ensure Agri_Sorter root is in sys.path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from app.contracts import (
    FruitDetection,
    FruitEvent,
    STATUS_FRESH,
    STATUS_REJECT,
    load_app_config,
)
from app.hardware_sorter import (
    HardwareCoordinatorThread,
    ServoActuator,
    UltrasonicSensor,
    create_hardware_coordinator_from_config,
)
from app.tracker import FruitTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main_sorter")


# ============================================================================
# Synthetic Conveyor Frame Generator
# ============================================================================

class SyntheticConveyorGenerator:
    """Generates synthetic video frames and fruit detections simulating a conveyor belt."""

    def __init__(
        self,
        width: int = 640,
        height: int = 640,
        fruit_interval_frames: int = 40,
    ):
        self.width = width
        self.height = height
        self.fruit_interval_frames = fruit_interval_frames
        self.frame_idx = 0

        # Active synthetic fruits traversing the conveyor
        # Each item: dict with {track_type, condition, y_pos, x_pos, size, speed}
        self.active_fruits: List[Dict[str, Any]] = []
        self._fruit_sequence = [
            ("apple", "fresh"),
            ("orange", "rotten"),
            ("banana", "bruised"),
            ("apple", "fresh"),
            ("mango", "fresh"),
            ("orange", "bug_bite"),
        ]
        self._seq_idx = 0

    def step(self) -> Tuple[np.ndarray, List[FruitDetection]]:
        """Advance simulation by 1 frame. Returns (frame, detections)."""
        self.frame_idx += 1

        # Spawn a new fruit periodically
        if self.frame_idx % self.fruit_interval_frames == 1:
            fruit_type, condition = self._fruit_sequence[self._seq_idx % len(self._fruit_sequence)]
            self._seq_idx += 1
            self.active_fruits.append({
                "fruit_type": fruit_type,
                "condition": condition,
                "x_pos": self.width // 2,
                "y_pos": 30.0,
                "radius": 35.0,
                "speed": 8.0,  # pixels per frame downward
            })

        # Draw frame (dark conveyor belt)
        frame = np.full((self.height, self.width, 3), 40, dtype=np.uint8)
        # Draw belt track lines
        cv2.line(frame, (80, 0), (80, self.height), (70, 70, 70), 2)
        cv2.line(frame, (self.width - 80, 0), (self.width - 80, self.height), (70, 70, 70), 2)

        # Draw egress threshold line
        egress_y = int(self.height * 0.85)
        cv2.line(frame, (0, egress_y), (self.width, egress_y), (0, 165, 255), 2)
        cv2.putText(
            frame,
            "CAMERA EGRESS LINE",
            (100, egress_y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 165, 255),
            1,
        )

        detections: List[FruitDetection] = []
        surviving_fruits: List[Dict[str, Any]] = []

        for fruit in self.active_fruits:
            fruit["y_pos"] += fruit["speed"]
            cx = int(fruit["x_pos"])
            cy = int(fruit["y_pos"])
            r = int(fruit["radius"])

            # Render fruit on synthetic frame
            is_defect = fruit["condition"] != "fresh"
            color = (0, 0, 220) if is_defect else (0, 200, 50)
            cv2.circle(frame, (cx, cy), r, color, -1)
            cv2.putText(
                frame,
                f"{fruit['fruit_type']} ({fruit['condition']})",
                (cx - r, cy - r - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )

            x1 = max(0, cx - r)
            y1 = max(0, cy - r)
            x2 = min(self.width, cx + r)
            y2 = min(self.height, cy + r)

            # Produce detection while still inside frame
            if y1 < self.height:
                detections.append(
                    FruitDetection(
                        bbox=(x1, y1, x2, y2),
                        confidence=0.92,
                        class_id=0 if not is_defect else 1,
                        class_name=f"{fruit['fruit_type']}_{fruit['condition']}",
                        fruit_type=fruit["fruit_type"],
                        condition=fruit["condition"],
                        is_defect=is_defect,
                    )
                )

            # Keep fruit until it has completely traversed past frame bottom
            if y1 < self.height + 50:
                surviving_fruits.append(fruit)

        self.active_fruits = surviving_fruits
        return frame, detections


# ============================================================================
# Main Sorter Pipeline Runner
# ============================================================================

def run_sorter(
    simulate: bool = True,
    video_path: Optional[str] = None,
    config_path: Optional[str] = None,
    max_frames: int = 150,
    fps: float = 30.0,
    test_underrun: bool = True,
    headless: bool = True,
) -> Dict[str, Any]:
    """Execute the complete vision-tracking-hardware sorting pipeline."""
    cfg = load_app_config(config_path)
    logger.info("Configuration successfully loaded.")

    # 1. Initialize tracker
    tracker = FruitTracker(config=cfg)
    fruit_queue = tracker.get_fifo_queue()

    # 2. Initialize hardware coordinator thread
    event_log_path = cfg.get("logging", {}).get("event_log_path", "runtime/fruit_events.csv")
    coordinator = create_hardware_coordinator_from_config(
        cfg=cfg,
        fruit_queue=fruit_queue,
        simulate=simulate,
        event_log_path=event_log_path,
    )

    # Start hardware coordinator background thread
    coordinator.start()
    logger.info(f"Hardware coordinator active (simulate={simulate}). Log: {event_log_path}")

    # Scheduled ultrasonic arrival queue for simulated conveyor transit
    # Maps future frame index -> fruit arrival at HC-SR04
    scheduled_triggers: List[int] = []
    # Egress travel time from camera line to HC-SR04 (e.g. 15 frames @ 30fps = 0.5s)
    transit_delay_frames = 15

    # 3. Ingestion source
    synthetic_gen = SyntheticConveyorGenerator()
    frame_count = 0
    start_time = time.time()

    try:
        while frame_count < max_frames:
            frame_count += 1
            current_time = time.time()

            # Ingest frame and detections
            frame, detections = synthetic_gen.step()
            frame_shape = (frame.shape[0], frame.shape[1])

            # Update tracker and detect camera egress
            queue_len_before = fruit_queue.qsize()
            active_tracks = tracker.update(
                detections=detections,
                frame_shape=frame_shape,
                timestamp=current_time,
            )
            queue_len_after = fruit_queue.qsize()

            # If a fruit newly egressed into the FIFO queue, schedule its HC-SR04 trigger
            if queue_len_after > queue_len_before:
                arrival_frame = frame_count + transit_delay_frames
                scheduled_triggers.append(arrival_frame)
                logger.info(
                    f"[Frame {frame_count:03d}] Fruit egressed camera view -> Scheduled at HC-SR04 on frame {arrival_frame}"
                )

            # Check if any scheduled fruit has arrived at the ultrasonic sensor
            while scheduled_triggers and scheduled_triggers[0] <= frame_count:
                scheduled_triggers.pop(0)
                logger.info(f"[Frame {frame_count:03d}] Fruit passing HC-SR04 -> Simulating ultrasonic distance dip (5.0 cm)")
                coordinator.ultrasonic.set_simulated_distance(5.0)
                # Let check_trigger fire or trigger directly
                coordinator.ultrasonic.check_trigger(current_time=current_time)
                # Reset distance back to idle
                coordinator.ultrasonic.set_simulated_distance(50.0)

            # Test Queue Underrun (untouched fruit reaching flap without camera detection)
            if test_underrun and frame_count == max_frames // 2:
                logger.info(
                    f"[Frame {frame_count:03d}] INJECTING TEST QUEUE UNDERRUN: Untracked object at flap with empty queue!"
                )
                coordinator.ultrasonic.set_simulated_distance(4.0)
                coordinator.ultrasonic.check_trigger(current_time=current_time)
                coordinator.ultrasonic.set_simulated_distance(50.0)

            # Console status print periodically
            if frame_count % 30 == 0:
                logger.info(
                    f"Progress: Frame {frame_count}/{max_frames} | Queue: {fruit_queue.qsize()} | "
                    f"Triggers: {coordinator.total_triggers} | Fresh: {coordinator.fresh_passed_count} | "
                    f"Rejected: {coordinator.defect_rejected_count} | Underrun: {coordinator.underrun_rejected_count} | "
                    f"Servo Angle: {coordinator.servo.angle}°"
                )

            # Display frame if not headless
            if not headless:
                cv2.imshow("Agri_Sorter Simulation", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # Frame rate throttle in simulate mode
            time.sleep(1.0 / max(1.0, fps))

        # Drain any remaining arrivals before stopping
        time.sleep(0.5)

    finally:
        logger.info("Stopping hardware coordinator thread...")
        coordinator.stop(timeout=2.0)
        if not headless:
            cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    summary = {
        "frames_processed": frame_count,
        "elapsed_seconds": round(elapsed, 2),
        "total_triggers": coordinator.total_triggers,
        "fresh_passed_count": coordinator.fresh_passed_count,
        "defect_rejected_count": coordinator.defect_rejected_count,
        "underrun_rejected_count": coordinator.underrun_rejected_count,
        "servo_rejection_strokes": coordinator.servo.rejection_count,
        "event_log_path": str(event_log_path),
    }

    logger.info("=" * 60)
    logger.info("AGRI_SORTER PIPELINE EXECUTION SUMMARY")
    logger.info("=" * 60)
    for k, v in summary.items():
        logger.info(f"  {k:28s}: {v}")
    logger.info("=" * 60)
    return summary


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Agri_Sorter Real-Time & Simulated Execution")
    parser.add_argument(
        "--simulate",
        action="store_true",
        default=True,
        help="Run in simulation mode with mock hardware and synthetic video (default: True)",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Optional path to input video file for playback",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config (defaults to config/config.yaml)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=120,
        help="Maximum frames to process in simulation (default: 120)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Simulation frame rate (default: 30.0)",
    )
    parser.add_argument(
        "--test-underrun",
        action="store_true",
        default=True,
        help="Inject an untracked fruit at midpoint to test fail-safe reject",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Run without GUI window (default: True)",
    )

    args = parser.parse_args()

    run_sorter(
        simulate=args.simulate,
        video_path=args.video,
        config_path=args.config,
        max_frames=args.frames,
        fps=args.fps,
        test_underrun=args.test_underrun,
        headless=args.headless,
    )


if __name__ == "__main__":
    main()
