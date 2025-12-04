# main.py

"""
Main pipeline entrypoint for the SDSE MLOps project.

Pipeline steps:
1. Data Processing
2. Model Training
3. Model Evaluation
4. Model Selection
5. Deployment

Each step writes artifacts into ./artifacts/ or ./deployment/.
"""

from data_processing import run_data_processing
from model_training import run_model_training
from model_evaluation import evaluate_model
from model_selection import select_best_model
from deploy import deploy_model


def main():

    print("\n==============================")
    print("STEP 1 — DATA PROCESSING")
    print("==============================")

    processed_df = run_data_processing()
    print(f"Data processing complete. Final dataset size: {processed_df.shape}")


    print("\n==============================")
    print("STEP 2 — MODEL TRAINING")
    print("==============================")

    # Train models separately (LR + XGBoost)
    # so we can compare them fairly
    print("\nTraining Logistic Regression...")
    lr_model, X_test_lr, y_test_lr = run_model_training(model_type="logreg")

    print("\nTraining XGBoost...")
    xgb_model, X_test_xgb, y_test_xgb = run_model_training(model_type="xgboost")


    print("\n==============================")
    print("STEP 3 — MODEL EVALUATION")
    print("==============================")

    print("\nEvaluating Logistic Regression...")
    lr_metrics = evaluate_model(lr_model, X_test_lr, y_test_lr, model_type="logreg")

    print("\nEvaluating XGBoost...")
    xgb_metrics = evaluate_model(xgb_model, X_test_xgb, y_test_xgb, model_type="xgboost")


    print("\n==============================")
    print("STEP 4 — MODEL SELECTION")
    print("==============================")

    best_model_type = select_best_model()
    print(f"Best model selected: {best_model_type.upper()}")


    print("\n==============================")
    print("STEP 5 — DEPLOYMENT")
    print("==============================")

    deploy_metadata = deploy_model()
    print("\nDeployment complete.")
    print("Deployment metadata:", deploy_metadata)


    print("\n==============================")
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("==============================\n")


if __name__ == "__main__":
    main()
