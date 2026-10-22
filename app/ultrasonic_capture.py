"""Command-line utility for ultrasonic data capture and optional inline prediction."""
import argparse
import csv
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.hardware.config import load_config
from app.hardware.pipeline import UltrasonicPipeline

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "hardware.toml"
STANDARD_FEATURES = [
    "travel_time_us",
    "received_amplitude",
    "fruit_diameter_mm",
    "weight_g",
    "temperature_c",
]


class RuleBasedAcousticClassifier:
    """Lightweight rule-based acoustic quality classifier.

    Evaluates 40 kHz ultrasonic ToF and amplitude against acoustic attenuation thresholds.
    """

    def __init__(
        self,
        amplitude_threshold: float = 45.0,
        min_travel_time_us: float = 2500.0,
        max_travel_time_us: float = 6500.0,
    ):
        self.amplitude_threshold = amplitude_threshold
        self.min_travel_time_us = min_travel_time_us
        self.max_travel_time_us = max_travel_time_us

    def predict(self, X: Any) -> List[int]:
        predictions: List[int] = []
        for row in X:
            if isinstance(row, dict):
                amp = float(row.get("received_amplitude", 0.0))
                tof = float(row.get("travel_time_us", 0.0))
            elif hasattr(row, "__getitem__"):
                try:
                    tof = float(row[0])
                    amp = float(row[1])
                except (IndexError, TypeError, ValueError):
                    amp = 0.0
                    tof = 0.0
            else:
                amp = 0.0
                tof = 0.0

            # 0 = healthy (first quality), 1 = defective (second quality)
            if amp < self.amplitude_threshold or tof < self.min_travel_time_us or tof > self.max_travel_time_us:
                predictions.append(1)
            else:
                predictions.append(0)
        return predictions

    @classmethod
    def from_dict(cls, data: dict) -> "RuleBasedAcousticClassifier":
        return cls(
            amplitude_threshold=float(data.get("amplitude_threshold", 45.0)),
            min_travel_time_us=float(data.get("min_travel_time_us", 2500.0)),
            max_travel_time_us=float(data.get("max_travel_time_us", 6500.0)),
        )

    def to_dict(self) -> dict:
        return {
            "type": "RuleBasedAcousticClassifier",
            "amplitude_threshold": self.amplitude_threshold,
            "min_travel_time_us": self.min_travel_time_us,
            "max_travel_time_us": self.max_travel_time_us,
        }


def load_prediction_model(model_path: str):
    """Load acoustic classifier model and its feature list from JSON file.

    Args:
        model_path: Path to json rule model file.

    Returns:
        Tuple[Any, List[str]]: (model_object, list_of_features)
    """
    path = Path(model_path)
    if path.suffix.lower() == ".json":
        import json
        with open(path, "r") as f:
            data = json.load(f)
        features = data.get("features", STANDARD_FEATURES)
        model = RuleBasedAcousticClassifier.from_dict(data)
        return model, features

    warnings.warn(f"Unsupported model format: {model_path}. Using lightweight RuleBasedAcousticClassifier fallback.")
    return RuleBasedAcousticClassifier(), STANDARD_FEATURES


def predict_row(
    model: Any,
    features: List[str],
    travel_time_us: float,
    received_amplitude: int,
    fruit_diameter_mm: Optional[float] = None,
    weight_g: Optional[float] = None,
    temperature_c: Optional[float] = None,
) -> int:
    """Predict internal label using acoustic rule evaluator."""
    if isinstance(model, dict):
        amp_thresh = float(model.get("amplitude_threshold", 45.0))
        return 0 if received_amplitude >= amp_thresh else 1

    feature_map = {
        "travel_time_us": float(travel_time_us),
        "received_amplitude": float(received_amplitude),
        "fruit_diameter_mm": float(fruit_diameter_mm) if fruit_diameter_mm is not None else float("nan"),
        "weight_g": float(weight_g) if weight_g is not None else float("nan"),
        "temperature_c": float(temperature_c) if temperature_c is not None else float("nan"),
    }

    row_vals = [[feature_map.get(k, float("nan")) for k in features]]

    if hasattr(model, "predict"):
        try:
            pred = model.predict(row_vals)[0]
            return int(pred)
        except Exception:
            return 0 if received_amplitude >= 45 else 1

    return 0 if received_amplitude >= 45 else 1


def run_capture(
    config_path: Optional[str] = None,
    count: int = 1,
    csv_out: Optional[str] = None,
    model_path: Optional[str] = None,
    mock: bool = False,
    delay: float = 0.1,
    diameter: Optional[float] = None,
    weight: Optional[float] = None,
    temperature: Optional[float] = None,
    use_mcu: bool = False,
    use_esp32: bool = False,
    use_arduino: bool = False,
) -> List[Dict[str, Any]]:
    """Execute ultrasonic measurement capture loop and write CSV output.

    Returns:
        List[Dict[str, Any]]: List of recorded measurement dictionaries.
    """
    use_mcu = bool(use_mcu or use_esp32 or use_arduino)
    cfg = load_config(config_path)
    model = None
    features = STANDARD_FEATURES

    if model_path:
        model, features = load_prediction_model(model_path)
        if diameter is None or weight is None or temperature is None:
            warnings.warn(
                "Auxiliary features (diameter, weight, temperature) not fully supplied. "
                "Predicting with NaNs may degrade classification accuracy."
            )

    fieldnames = ["travel_time_us", "received_amplitude"]
    if diameter is not None or use_mcu:
        fieldnames.append("fruit_diameter_mm")
    if weight is not None:
        fieldnames.append("weight_g")
    if temperature is not None:
        fieldnames.append("temperature_c")
    if model is not None:
        fieldnames.append("internal_label_pred")

    out_file = None
    if csv_out and csv_out != "-":
        out_path = Path(csv_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        file_is_new = not out_path.exists() or out_path.stat().st_size == 0
        out_file = open(out_path, "a", newline="")
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        if file_is_new:
            writer.writeheader()
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()

    captured_rows: List[Dict[str, Any]] = []

    try:
        if use_mcu:
            from app.hardware.arduino_bridge import ArduinoBridge

            bridge = ArduinoBridge(config=cfg, mock=mock)
            bridge.start()
            try:
                for i in range(count):
                    # In mock or hardware mode, ensure at least one reading is acquired
                    reading = bridge.get_latest_reading()
                    if reading is None:
                        t_wait = time.time()
                        while reading is None and (time.time() - t_wait) < 0.2:
                            time.sleep(0.01)
                            reading = bridge.get_latest_reading()

                    tof = float(reading.tof_us) if reading is not None else 4210.0
                    amp = int(reading.dist_mm) if reading is not None else 72
                    measured_diameter = (
                        float(reading.fruit_diameter_mm)
                        if reading is not None and reading.fruit_diameter_mm is not None
                        else 0.0
                    )
                    effective_diameter = diameter if diameter is not None else measured_diameter

                    row: Dict[str, Any] = {
                        "travel_time_us": tof,
                        "received_amplitude": amp,
                        "fruit_diameter_mm": effective_diameter,
                    }

                    if weight is not None:
                        row["weight_g"] = weight
                    if temperature is not None:
                        row["temperature_c"] = temperature

                    if model is not None:
                        pred = predict_row(
                            model=model,
                            features=features,
                            travel_time_us=tof,
                            received_amplitude=amp,
                            fruit_diameter_mm=effective_diameter,
                            weight_g=weight,
                            temperature_c=temperature,
                        )
                        row["internal_label_pred"] = pred

                    writer.writerow(row)
                    if out_file:
                        out_file.flush()
                    captured_rows.append(row)

                    if delay > 0 and i < count - 1:
                        time.sleep(delay)
            finally:
                bridge.stop()
        else:
            with UltrasonicPipeline(config=cfg, mock=mock) as pipeline:
                for i in range(count):
                    meas = pipeline.measure()
                    row: Dict[str, Any] = {
                        "travel_time_us": meas["travel_time_us"],
                        "received_amplitude": meas["received_amplitude"],
                    }

                    if diameter is not None:
                        row["fruit_diameter_mm"] = diameter
                    if weight is not None:
                        row["weight_g"] = weight
                    if temperature is not None:
                        row["temperature_c"] = temperature

                    if model is not None:
                        pred = predict_row(
                            model=model,
                            features=features,
                            travel_time_us=meas["travel_time_us"],
                            received_amplitude=meas["received_amplitude"],
                            fruit_diameter_mm=diameter,
                            weight_g=weight,
                            temperature_c=temperature,
                        )
                        row["internal_label_pred"] = pred

                    writer.writerow(row)
                    if out_file:
                        out_file.flush()
                    captured_rows.append(row)

                    if delay > 0 and i < count - 1:
                        time.sleep(delay)
    finally:
        if out_file:
            out_file.close()

    return captured_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture ultrasonic sensor data for fruit quality assessment.")
    parser.add_argument("--config", default=None, help="Path to hardware.toml configuration file.")
    parser.add_argument("--count", type=int, default=1, help="Number of measurements to capture.")
    parser.add_argument("--csv-out", default=None, help="Output CSV path (default: stdout).")
    parser.add_argument("--model", default=None, help="Path to acoustic rule configuration (.json).")
    parser.add_argument("--mock", action="store_true", help="Run with mock hardware drivers.")
    parser.add_argument("--delay", type=float, default=0.1, help="Delay in seconds between measurements.")
    parser.add_argument("--diameter", type=float, default=None, help="Fruit diameter in mm (optional auxiliary feature).")
    parser.add_argument("--weight", type=float, default=None, help="Fruit weight in grams (optional auxiliary feature).")
    parser.add_argument("--temperature", type=float, default=None, help="Ambient temperature in °C (optional auxiliary feature).")
    parser.add_argument("--use-arduino", action="store_true", help="Read ultrasonic samples from Arduino Uno USB serial bridge.")
    parser.add_argument("--use-mcu", action="store_true", help="Alias for --use-arduino (microcontroller USB serial bridge).")
    parser.add_argument("--use-esp32", action="store_true", help="Alias for --use-arduino.")

    args = parser.parse_args()

    run_capture(
        config_path=args.config,
        count=args.count,
        csv_out=args.csv_out,
        model_path=args.model,
        mock=args.mock,
        delay=args.delay,
        diameter=args.diameter,
        weight=args.weight,
        temperature=args.temperature,
        use_arduino=bool(args.use_arduino or args.use_mcu or args.use_esp32),
    )


if __name__ == "__main__":
    main()
