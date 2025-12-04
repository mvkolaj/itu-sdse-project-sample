# model_inference.py

import json
import pandas as pd
import joblib
from pathlib import Path
from xgboost import XGBRFClassifier


ARTIFACTS_DIR = Path("artifacts")
MODEL_LR_PATH = ARTIFACTS_DIR / "lead_model_lr.pkl"
MODEL_XGB_PATH = ARTIFACTS_DIR / "lead_model_xgboost.json"
COLUMNS_PATH = ARTIFACTS_DIR / "columns_list.json"


# ------------------------------------------------------------
# Load required columns (order matters)
# ------------------------------------------------------------
def load_column_order():
    """
    Loads the exact column order used during training.
    """
    if not COLUMNS_PATH.exists():
        raise FileNotFoundError("Missing artifacts/columns_list.json")

    with open(COLUMNS_PATH) as f:
        return json.load(f)["column_names"]


# ------------------------------------------------------------
# Load a trained model
# ------------------------------------------------------------
def load_model(model_type="logreg"):
    """
    Load the saved model based on model_type.
    model_type: 'logreg' or 'xgboost'
    """

    if model_type == "xgboost":
        if not MODEL_XGB_PATH.exists():
            raise FileNotFoundError("Missing XGBoost model .json file")

        model = XGBRFClassifier()
        model.load_model(str(MODEL_XGB_PATH))
        return model

    # Logistic Regression
    if not MODEL_LR_PATH.exists():
        raise FileNotFoundError("Missing Logistic Regression model .pkl file")

    return joblib.load(MODEL_LR_PATH)


# ------------------------------------------------------------
# Prepare new data for inference
# ------------------------------------------------------------
def prepare_input(df: pd.DataFrame, required_cols):
    """
    Ensures new data matches the training feature set exactly:
    - adds missing columns (filled with 0)
    - removes extra columns
    - orders columns identically to training
    """

    # Add missing columns as zeros
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0

    # Remove columns not used during training
    df = df[required_cols]

    # Convert everything to float (models expect this)
    df = df.astype("float64")

    return df


# ------------------------------------------------------------
# Predict classes or probabilities
# ------------------------------------------------------------
def predict(input_csv, model_type="logreg", return_proba=False):
    """
    Main inference function.
    Loads model, loads required columns, formats input, and predicts.

    Parameters:
        input_csv: path to the new data CSV
        model_type: "logreg" or "xgboost"
        return_proba: bool → return probability instead of class label

    Returns:
        predictions (array)
    """

    # Load the trained model
    model = load_model(model_type)

    # Load required column order
    required_cols = load_column_order()

    # Load new data
    df = pd.read_csv(input_csv)

    # Format & align with training data
    df_prepared = prepare_input(df, required_cols)

    # Predict
    if return_proba:
        try:
            preds = model.predict_proba(df_prepared)[:, 1]
        except:
            preds = model.predict(df_prepared)
    else:
        preds = model.predict(df_prepared)

    print(f"\nPredictions ({model_type}):", preds)

    return preds
