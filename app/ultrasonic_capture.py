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


def load_prediction_model(model_path: str):
    """Load trained classifier model and its feature list from joblib file.

    Args:
        model_path: Path to joblib file.

    Returns:
        Tuple[Any, List[str]]: (model_object, list_of_features)
    """
    try:
        import joblib
    except ImportError:
        raise ImportError("joblib is required to load models. Install with: pip install joblib")

    payload = joblib.load(model_path)
    if isinstance(payload, dict) and "model" in payload:
        model = payload["model"]
        features = payload.get("features", STANDARD_FEATURES)
    else:
        model = payload
        features = STANDARD_FEATURES

    return model, features


def predict_row(
    model: Any,
    features: List[str],
    travel_time_us: float,
    received_amplitude: int,
    fruit_diameter_mm: Optional[float] = None,
    weight_g: Optional[float] = None,
    temperature_c: Optional[float] = None,
) -> int:
    """Predict internal label using loaded model."""
    feature_map = {
        "travel_time_us": float(travel_time_us),
        "received_amplitude": float(received_amplitude),
        "fruit_diameter_mm": float(fruit_diameter_mm) if fruit_diameter_mm is not None else float("nan"),
        "weight_g": float(weight_g) if weight_g is not None else float("nan"),
        "temperature_c": float(temperature_c) if temperature_c is not None else float("nan"),
    }

    try:
        import pandas as pd
        df = pd.DataFrame([{k: feature_map.get(k, float("nan")) for k in features}])[features]
        pred = model.predict(df)[0]
    except Exception:
        row_vals = [[feature_map.get(k, float("nan")) for k in features]]
        pred = model.predict(row_vals)[0]

    return int(pred)


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
) -> List[Dict[str, Any]]:
    """Execute ultrasonic measurement capture loop and write CSV output.

    Returns:
        List[Dict[str, Any]]: List of recorded measurement dictionaries.
    """
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
    if diameter is not None:
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
    parser.add_argument("--model", default=None, help="Path to trained XGBoost internal model (.joblib).")
    parser.add_argument("--mock", action="store_true", help="Run with mock hardware drivers.")
    parser.add_argument("--delay", type=float, default=0.1, help="Delay in seconds between measurements.")
    parser.add_argument("--diameter", type=float, default=None, help="Fruit diameter in mm (optional auxiliary feature).")
    parser.add_argument("--weight", type=float, default=None, help="Fruit weight in grams (optional auxiliary feature).")
    parser.add_argument("--temperature", type=float, default=None, help="Ambient temperature in °C (optional auxiliary feature).")

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
    )


if __name__ == "__main__":
    main()
