# deploy.py

import json
import shutil
from pathlib import Path

ARTIFACTS_DIR = Path("artifacts")
DEPLOY_DIR = Path("deployment")

# Model artifact paths
LR_MODEL = ARTIFACTS_DIR / "lead_model_lr.pkl"
XGB_MODEL = ARTIFACTS_DIR / "lead_model_xgboost.json"
SELECTED_MODEL_FILE = ARTIFACTS_DIR / "selected_model.json"


def load_selected_model_type() -> str:
    """
    Reads artifacts/selected_model.json and retrieves the model type.
    """
    if not SELECTED_MODEL_FILE.exists():
        raise FileNotFoundError(
            "Model selection file not found. Run model_selection.select_best_model() first."
        )

    with SELECTED_MODEL_FILE.open("r") as f:
        data = json.load(f)

    return data["selected_model"]


def deploy_model():
    """
    Copies the best model into a deployment folder and writes deployment metadata.
    """
    DEPLOY_DIR.mkdir(exist_ok=True)

    model_type = load_selected_model_type()

    if model_type == "logreg":
        model_src = LR_MODEL
        model_dest = DEPLOY_DIR / "model.pkl"
    else:
        model_src = XGB_MODEL
        model_dest = DEPLOY_DIR / "model.json"

    if not model_src.exists():
        raise FileNotFoundError(f"Model file not found: {model_src}")

    # Copy model to deployment folder
    shutil.copy(model_src, model_dest)

    # Save metadata
    metadata = {
        "model_type": model_type,
        "source_path": str(model_src),
        "deployment_path": str(model_dest),
        "ready": True,
    }

    with open(DEPLOY_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"\n=== MODEL DEPLOYED SUCCESSFULLY ===")
    print(f"Selected model: {model_type}")
    print(f"Copied to: {model_dest}")
    print("Metadata saved to deployment/metadata.json")

    return metadata


if __name__ == "__main__":
    deploy_model()
