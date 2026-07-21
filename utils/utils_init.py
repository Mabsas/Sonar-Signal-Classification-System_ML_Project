"""
utils package
--------------
ML logic for the Sonar Signal Classification System, kept separate
from the Streamlit UI in app.py.

Modules:
    prediction.py         Single-sample inference (predict_sonar)
    batch_prediction.py    CSV validation and bulk inference
"""

from utils.prediction import predict_sonar
from utils.batch_prediction import validate_csv, predict_batch, normalize_uploaded_dataframe

__all__ = ["predict_sonar", "validate_csv", "predict_batch", "normalize_uploaded_dataframe"]
