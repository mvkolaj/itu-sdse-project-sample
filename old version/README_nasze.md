# ITU SDSE Project – ML Pipeline with Dagger & GitHub Actions

## Project Overview
This project implements a **machine learning pipeline** to identify **new possible customers** on a website.  
It uses **user behavior data** as input, and the target is whether the user converted into a customer (classification problem).

The ML pipeline handles **data preprocessing, feature engineering, model training and evaluation, model selection and deployment**.  
The original notebook has been refactored into **Python scripts**, and the pipeline is containerized with **Dagger** and automated via **GitHub Actions** for reproducibility.

> Note: The starting notebook may contain extra comments, markdown, and unused code. The goal is to refactor it into clean Python scripts while keeping functionality identical. NIE WIEM CZY TO MA TU BYC W SUMIE   

---

## Repository Structure
The repository is organized as follows:
.
├── .dvc/                                       # DVC internal files
├── .github/workflows/
│   └── test_action.yml                         # GitHub Actions CI workflow
│
├── data/
│   └── raw_data.csv.dvc                        # DVC pointer to raw dataset
│
├── docs/                                       # Architecture diagrams and documentation assets
│
├── go/
│   ├── pipeline.go                             # Dagger-based pipeline 
│   └── go.mod                                  # Go module definition
│
├── notebooks/                                  # Exploratory notebooks and experiments
│
├── src/                                        # Core Python source code
│   ├── __init__.py
│   ├── data_preprocessing.py                   # Data loading and preprocessing
│   ├── data_features.py                        # Feature engineering
│   ├── model_training.py                       # Model training
│   └── model_evaluation_and_deployment.py      # Evaluation, model selection and deployment
│
├── README.md                                   # Project documentation
├── requirements.txt                            # Python dependencies




---
## How to Run Locally 



---
## Using GitHub Actions Workflow


---
## Notes 


---

## Authors
Zofia Brodewicz (zobr@itu.dk) and Mikolaj Andrzejewski (mikoa@itu.dk)
