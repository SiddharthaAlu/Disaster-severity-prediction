# 🌍 Disaster Severity Prediction System

## 📌 Project Overview

The Disaster Severity Prediction System is a Machine Learning application that predicts whether a disaster event will be classified as a **Major Disaster** or **Non-Major Disaster** based on historical disaster data.

The project follows a complete Machine Learning workflow, including data preprocessing, feature engineering, feature selection, model training, evaluation, and deployment using Streamlit.

---

## 🎯 Objective

To build a predictive model that can identify major disaster events using key disaster-related factors and assist in disaster management and response planning.

---

## 📂 Dataset

**Dataset:** `synthetic_disaster_events_2025.csv`

The dataset contains information about disaster events such as:

- Disaster Type
- Location
- Severity Level
- Affected Population
- Estimated Economic Loss
- Infrastructure Damage Index
- Response Time
- Aid Provided
- Major Disaster Status (Target Variable)

---

## ⚙️ Machine Learning Workflow

1. Data Loading
2. Exploratory Data Analysis (EDA)
3. Data Preprocessing
4. Feature Engineering
5. Feature Selection
6. Train-Test Split
7. Feature Scaling
8. Model Training using Logistic Regression
9. Model Evaluation
10. Model Deployment using Streamlit

---

## 🔍 Selected Features

The final model was trained using the following features:

- Severity Level
- Affected Population
- Estimated Economic Loss (USD)
- Infrastructure Damage Index

---

## 🤖 Model Used

**Logistic Regression**

Reason for Selection:

- Simple and interpretable
- Fast training and prediction
- Excellent performance on the dataset
- Suitable for binary classification problems

---

## 📊 Evaluation Metrics

The model was evaluated using:

- Accuracy
- F1 Score

### Results

| Metric | Score |
|----------|----------|
| Accuracy | 99.95% |
| F1 Score | 99.94% |

---

## 🚀 Deployment

The trained model is deployed using **Streamlit**.

The application allows users to input:

- Severity Level
- Affected Population
- Estimated Economic Loss
- Infrastructure Damage Index

and receive a prediction indicating whether the disaster is likely to be a Major Disaster or a Non-Major Disaster.

---

## 📁 Project Structure

```text
Disaster Severity prediction/
│
├── data/
│   └── synthetic_disaster_events_2025.csv
│
├── models/
│   ├── disaster_model.pkl
│   └── scaler.pkl
│
├── train_model.py
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── venv/
```

---

## ▶️ How to Run

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train the Model

```bash
python train_model.py
```

### Run the Streamlit Application

```bash
streamlit run app.py
```

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Joblib
- Streamlit

---

## 📌 Conclusion

A Disaster Severity Prediction System was successfully developed using Logistic Regression. The model achieved high predictive performance with an Accuracy of 99.95% and an F1 Score of 99.94%. The final model was deployed as an interactive Streamlit application for real-time disaster severity prediction.