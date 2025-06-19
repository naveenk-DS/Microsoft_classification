# 🛡️ Microsoft - Classifying Cybersecurity Incidents
This project focuses on classifying cybersecurity incidents using machine learning techniques. The goal is to build a model that can identify the type of incident based on input features, helping security teams respond more effectively.

# 📌 Project Overview
Cybersecurity threats are a growing concern. This project uses a dataset simulating Microsoft-like security logs to predict the category of a cybersecurity incident (e.g., Malware, Phishing, Ransomware, etc.).

## 📁 Dataset
Source: Simulated or anonymized Microsoft security incident logs.

## Features:

incident_id: Unique identifier

timestamp: Time of incident

source_ip, destination_ip

threat_level, protocol, port, user_action

description: Log summary

Target: incident_type (e.g., Malware, Phishing, Ransomware, etc.)

# ⚙️ Requirements
Install required libraries using:

bash
Copy
Edit
pip install -r requirements.txt
requirements.txt
txt
Copy
Edit
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
nltk
# 🧪 Key Steps
Data Cleaning & Preprocessing

Handle missing values

Convert timestamps

Encode categorical variables

Text preprocessing on descriptions

Exploratory Data Analysis (EDA)

Incident frequency

IP and Port analysis

Threat level distribution

Feature Engineering

Extract time-based features

NLP features from description using TF-IDF or CountVectorizer

Model Training

Models used: Random Forest, XGBoost, Logistic Regression

Evaluation metrics: Accuracy, Precision, Recall, F1 Score

Model Selection

Cross-validation

Confusion matrix visualization

# 📊 Results
Best model: XGBoost

Accuracy: XX%

F1 Score: XX%

Confusion Matrix: Included in /output/
