# CodeAlpha_CreditScoring

## Project Overview
This repository contains the implementation of a **Credit Scoring Model** developed as part of Task 1 for the CodeAlpha Machine Learning Internship 2025. The project aims to predict the creditworthiness of individuals (approve or deny credit applications) using the UCI Credit Approval dataset, which includes 690 samples and 15 anonymized features such as income, debt ratios, and payment history.

## Key Features
- **Data Preprocessing**: Loaded the UCI dataset, handled missing values with median imputation, and encoded categorical variables using LabelEncoder.
- **Exploratory Data Analysis (EDA)**: Conducted statistical analysis and generated visualizations (target distribution, correlation heatmap) to explore data patterns, saved in `eda_results/`.
- **Model Training**: Implemented and evaluated three classification algorithms:
  - **Logistic Regression**: Accuracy 84.06%, ROC-AUC 88.40%
  - **Decision Tree**: Accuracy 77.54%, ROC-AUC 77.61%
  - **Random Forest**: Accuracy 86.23%, ROC-AUC 91.28% (best model)
- **Model Selection**: Random Forest was selected as the best model based on ROC-AUC and saved as `credit_model.pkl`.
- **Prediction**: Developed a script to predict credit approval for new inputs, e.g., a sample input yielded a 7.00% approval probability.

## Project Structure
- `data/`: Contains the raw `credit.csv` dataset.
- `models/`: Stores processed data (`processed_credit_data.pkl`) and the trained model (`credit_model.pkl`).
- `src/`: Includes Python scripts:
  - `data_load.py`: Loads and preprocesses the dataset.
  - `eda.py`: Performs EDA with visualizations.
  - `model_train.py`: Trains and evaluates models.
  - `predict.py`: Makes predictions using the saved model.
- `eda_results/`: Contains generated plots (`target_distribution.png`, `correlation_heatmap.png`).

## Usage
1. **Clone the repository**:
2. **Install dependencies**:

3. **Run the scripts in order**:
- `python src/data_load.py` (preprocesses data)
- `python src/eda.py` (generates EDA visualizations)
- `python src/model_train.py` (trains and saves the best model)
- `python src/predict.py` (tests a sample prediction)

## Acknowledgments
A huge thank you to **@CodeAlpha** for providing this incredible learning opportunity.
## Tags
#MachineLearning #CreditScoring #CodeAlpha #Internship #DataScience
