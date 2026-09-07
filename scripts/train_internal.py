"""Train XGBoost on labelled ultrasonic/physical measurements."""
import argparse
from pathlib import Path
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

FEATURES = ["travel_time_us", "received_amplitude", "fruit_diameter_mm",
            "weight_g", "temperature_c"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--output", default="models/internal/xgb_internal.joblib")
    args = p.parse_args()
    df = pd.read_csv(args.csv)
    missing = [c for c in FEATURES + ["internal_label"] if c not in df]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = df.dropna(subset=FEATURES + ["internal_label"])
    X, y = df[FEATURES], df["internal_label"].astype(int)
    if y.nunique() < 2:
        raise ValueError("Need both healthy (0) and damaged (1) labelled fruit.")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    model = XGBClassifier(n_estimators=250, max_depth=4, learning_rate=.05,
                          subsample=.85, colsample_bytree=.85,
                          eval_metric="logloss", random_state=42)
    model.fit(Xtr, ytr)
    print(classification_report(yte, model.predict(Xte), zero_division=0))
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES}, path)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
