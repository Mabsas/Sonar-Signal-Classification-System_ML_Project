import joblib
import numpy as np
import pandas as pd

# Load saved artifacts
from xgboost import XGBClassifier

model = XGBClassifier()
model.load_model("artifacts/tuned_xgboost_model.json")
feature_order = joblib.load("artifacts/feature_order.pkl")
label_encoder = joblib.load("artifacts/label_encoder.pkl")


def predict_sonar(features):
    """
    Predict whether the sonar signal represents a Rock or a Mine.

    Parameters:
        features (list): List of 60 sonar feature values.

    Returns:
        tuple:
            prediction (str)
            confidence (float)
    """

    # Ensure exactly 60 features are provided
    if len(features) != len(feature_order):
        raise ValueError(
            f"Expected {len(feature_order)} features, but received {len(features)}."
        )

    # Ensure every value is numeric
    try:
        features = [float(value) for value in features]
    except (TypeError, ValueError) as exc:
        raise ValueError("All sonar feature values must be numeric.") from exc

    try:
        # Convert input to DataFrame with correct feature names
        input_df = pd.DataFrame([features], columns=feature_order)

        # Predict class
        prediction_encoded = model.predict(input_df)[0]

        # Convert prediction back to original label
        prediction = label_encoder.inverse_transform([prediction_encoded])[0]

        # Prediction confidence
        probabilities = model.predict_proba(input_df)[0]
        confidence = np.max(probabilities) * 100

    except Exception as exc:
        # Surface a clean, actionable error instead of a raw stack trace
        raise RuntimeError(f"Prediction failed: {exc}") from exc

    return prediction, confidence