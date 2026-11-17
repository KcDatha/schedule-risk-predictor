"""Shared prediction logic for the Schedule Risk Predictor.

Loads the trained artifacts from model/ and scores one project snapshot.
Used by the FastAPI app and by the smoke test, so both exercise exactly
the same code path.
"""
import json
import os

import joblib
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "model", "metrics.json")) as f:
    METRICS = json.load(f)
FEATURES = METRICS["features"]
CLS = joblib.load(os.path.join(HERE, "model", "slip_classifier.joblib"))
REG = joblib.load(os.path.join(HERE, "model", "delay_regressor.joblib"))


def predict(snapshot: dict) -> dict:
    row = pd.DataFrame([{k: float(snapshot[k]) for k in FEATURES}], columns=FEATURES)
    prob = float(CLS.predict_proba(row)[0, 1])
    delay = float(REG.predict(row)[0])
    band = "high" if prob >= 0.66 else "medium" if prob >= 0.4 else "low"
    return {
        "slip_probability": round(prob, 3),
        "predicted_delay_days": round(delay, 1),
        "risk_band": band,
        "top_drivers": [name for name, _ in METRICS["top_features_for_slip"][:3]],
        "model_notes": "Trained on a documented synthetic dataset built from earned value relationships. Demonstration model, not production data.",
    }
