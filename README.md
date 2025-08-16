# 🛡️ Classifying Cybersecurity Incidents with Machine Learning
## 📌 Project Overview

This project aims to classify cybersecurity incidents into categories using machine learning.
We preprocess raw security incident data, perform EDA (Exploratory Data Analysis), handle missing values, encode categorical features, and build ML models such as Random Forest, XGBoost, and Logistic Regression.

The goal is to predict the Category of incidents (e.g., TP, FP, BP) to help security analysts prioritize responses.

## 📂 Dataset

### The dataset consists of two files:

GUIDE_Train.csv → Training dataset

GUIDE_Test.csv → Testing dataset

## Key Columns:

IncidentId – Unique ID for incidents

AlertTitle – Title/description of alert

Category – Target variable (TP, FP, BP, etc.)

IncidentGrade, EntityType, ResourceType, etc. – Features used for classification

## ⚙️ Steps in the Project
### 1. Data Preprocessing

Removed duplicate/unnecessary columns (OrgId, AlertId, etc.).

Handled missing values:

Numerical columns → filled with mean

Categorical columns → filled with mode

Label Encoding used for categorical features.

### 2. Exploratory Data Analysis (EDA)

Distribution plots for categorical features.

Correlation heatmap (numerical features).

Feature importance analysis using Random Forest.

### 3. Model Selection & Training

Models tested:

Logistic Regression

Random Forest Classifier

XGBoost Classifier

### 4. Model Evaluation

Metrics used:

Accuracy

Precision, Recall, F1-Score (Macro-F1 for balanced evaluation)

Confusion Matrix

### 5. Model Tuning

Hyperparameter tuning using GridSearchCV / RandomizedSearchCV.

Class imbalance handled with SMOTE and class weights.

### 6. Final Model & Predictions

Best model selected (Random Forest / XGBoost).

Predictions saved into a CSV file (Predictions.csv).

## 🧪 Usage
### Install Dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn

### Run Training Script
python train.py

### Save Predictions
import pandas as pd

y_pred = model.predict(X_test)

pred_df = pd.DataFrame({
    "IncidentId": Test["IncidentId"],
    "Predicted_Category": y_pred
})

pred_df.to_csv("Predictions.csv", index=False)
print("✅ Predictions saved successfully!")

## Save Trained Model
import joblib
joblib.dump(model, "final_model.pkl")

Load Saved Model
model = joblib.load("final_model.pkl")

## 📊 Results

Best performing model: Random Forest (after tuning)

Macro-F1 Score: ~0.89 (example)

Balanced performance across TP, FP, BP categories

## 🔮 Future Improvements

Use deep learning (LSTM/Transformers) for better text-based features.

Add feature engineering from AlertTitle (NLP-based embeddings).

Deploy as a Streamlit dashboard for analysts.

# 👨‍💻 Author

## Naveen
Data Science Student @ GUVI – IIT Madras
