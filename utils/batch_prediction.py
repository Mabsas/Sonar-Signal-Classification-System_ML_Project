"""
Batch Prediction Module
------------------------
Handles CSV validation and bulk inference for the Sonar Signal
Classification System. Kept separate from prediction.py so that
single-sample and batch workflows can evolve independently, and
separate from app.py so the UI layer stays free of ML/validation logic.
"""

# ----------------------------------
# Load Libraries
# ----------------------------------
import pandas as pd

from utils.prediction import model, feature_order, label_encoder

# ----------------------------------
# Constants
# ----------------------------------
REQUIRED_COLUMNS = feature_order
REQUIRED_COLUMN_COUNT = len(feature_order)

LABEL_DISPLAY_MAP = {
    "R": "Rock",
    "M": "Mine",
}


# ----------------------------------
# Helper Functions
# ----------------------------------
def _looks_like_headerless_data(df: pd.DataFrame) -> bool:
    """
    Detect the common case where a raw, header-less dataset (like the
    original UCI Sonar CSV) was uploaded. When there's no header row,
    pandas mistakenly treats the first data row as column names - so
    the column names themselves look like numeric feature values
    (e.g. '0.02', '0.0371', ...) rather than 'Signal_1', 'Signal_2', etc.
    """

    if df.shape[1] != REQUIRED_COLUMN_COUNT:
        return False

    try:
        [float(col) for col in df.columns]
        return True
    except (TypeError, ValueError):
        return False


def normalize_uploaded_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Best-effort cleanup of an uploaded dataframe before validation, so
    common real-world formatting issues don't require the user to
    manually edit their file.

    Handles:
        1. Header-less files where the first row was misread as column
           names - the misread row is restored as real data and proper
           feature names are applied.
        2. Files with the right number of columns but different/misordered
           names - columns are renamed positionally to match the model's
           expected feature order.

    Parameters:
        df (pd.DataFrame): The raw uploaded dataframe.

    Returns:
        tuple:
            df (pd.DataFrame): The normalized dataframe (same object if
                no changes were needed).
            note (str): A user-facing explanation of what was changed,
                or an empty string if nothing was changed.
    """

    if df.shape[1] != REQUIRED_COLUMN_COUNT:
        return df, ""

    if list(df.columns) == REQUIRED_COLUMNS:
        return df, ""

    if _looks_like_headerless_data(df):
        # The "column names" are actually the first data row - restore it
        restored_row = pd.DataFrame([df.columns.tolist()], columns=REQUIRED_COLUMNS)
        df = df.copy()
        df.columns = REQUIRED_COLUMNS
        df = pd.concat([restored_row, df], ignore_index=True)
        return df, (
            "ℹ️ This file had no header row, so the first row was being "
            "read as column names. It's been restored as a data row and "
            "the standard sonar feature names were applied automatically."
        )

    # Right column count, wrong/misordered names - map positionally
    df = df.copy()
    df.columns = REQUIRED_COLUMNS
    return df, (
        "ℹ️ Column names didn't match the expected feature names, so "
        "they were mapped positionally (1st column → Signal_1, 2nd → "
        "Signal_2, etc.). Please double-check your column order is correct."
    )


def validate_csv(df: pd.DataFrame) -> tuple[bool, str]:
    """
    Validate an uploaded CSV against the requirements of the sonar model.

    Checks performed:
        1. File is not empty
        2. Column count matches the expected 60 features
        3. Column names exactly match the trained feature order
        4. No missing values are present
        5. All values are numeric

    Parameters:
        df (pd.DataFrame): The uploaded CSV, already read into memory.

    Returns:
        tuple:
            is_valid (bool): True if the CSV passes every check.
            message (str): Empty string if valid, otherwise a
                user-friendly description of the first failure.
    """

    if df.empty:
        return False, "The uploaded CSV file is empty."

    if df.shape[1] != REQUIRED_COLUMN_COUNT:
        return False, (
            f"CSV must contain exactly {REQUIRED_COLUMN_COUNT} feature "
            f"columns, but {df.shape[1]} were found."
        )

    if list(df.columns) != REQUIRED_COLUMNS:
        return False, (
            "CSV column names do not match the expected sonar feature "
            "names or their order. Please use the provided sample "
            "template in sample_data/sample_batch.csv."
        )

    if df.isnull().sum().sum() > 0:
        return False, "CSV contains missing values. Please clean the data and try again."

    try:
        df.astype(float)
    except ValueError:
        return False, "CSV contains non-numeric values in one or more feature columns."

    return True, ""


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the tuned XGBoost model against every row of a validated dataframe.

    Parameters:
        df (pd.DataFrame): A CSV dataframe that has already passed
            validate_csv().

    Returns:
        pd.DataFrame: The original data with two additional columns,
            'Prediction' (Rock/Mine) and 'Confidence (%)'.
    """

    numeric_df = df.astype(float)[REQUIRED_COLUMNS]

    encoded_predictions = model.predict(numeric_df)
    probabilities = model.predict_proba(numeric_df)

    predictions = label_encoder.inverse_transform(encoded_predictions)
    confidences = probabilities.max(axis=1) * 100

    results = df.copy()
    results["Prediction"] = [LABEL_DISPLAY_MAP.get(p, p) for p in predictions]
    results["Confidence (%)"] = confidences.round(2)

    return results
