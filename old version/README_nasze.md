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
├── README.md                <- Project overview and usage instructions
├── requirements.txt         <- Python dependencies
├── go
│   ├── pipeline.go          <- Dagger-based pipeline 
│   └── go.mod               <- Go module definition
│
├── data
│   └── raw_data.csv.dvc     <-DVC metadata file tracking the raw dataset stored in a remote data source
│
├── notebooks                <- Exploratory notebooks and experiments
│
├── docs                     <- Architecture diagrams and documentation assets
│
└── src                      <- Core Python source code
    ├── __init__.py
    ├── data_preprocessing.py <- Data loading and preprocessing
    ├── data_features.py      <- Feature engineering
    ├── model_training.py     <- Model training and hyperparameter tuning
    └── model_evaluation.py   <- Model evaluation and metrics



---
## How to Run Locally / How to Run the Project and Generate the Model Artifact



---
## Using GitHub Actions Workflow


---
## Notes 


---

## Authors
Zofia Brodewicz (zobr@itu.dk) and Mikolaj Andrzejewski (mikoa@itu.dk)
