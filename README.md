# Loan Default Risk Prediction

> **An end-to-end data science project to predict loan defaults using exploratory data analysis, feature engineering, statistical insights, machine learning modeling, and deployment-ready workflows.**

![Loan Default Risk](https://raw.githubusercontent.com/Samson-Chigbu/Loan_Default_Risk_Prediction/main/images/loan_default_risk.png)

---

## 📌 Project Overview

Predicting loan defaults is crucial for financial institutions to **reduce credit risk**, **improve approval processes**, and **enhance portfolio quality**.  
This project demonstrates a **complete Data Science workflow** — from raw data preprocessing and exploratory data analysis (EDA) to feature engineering, model building, evaluation, and interpretation — for predicting **loan default risk**.  
The goal is to help financial institutions identify high-risk borrowers early, enabling proactive risk management and reducing potential losses.


---

## 🎯 Problem Statement

Develop a robust machine learning pipeline to predict loan default risk, enabling better credit decisions and minimizing financial losses.  

## 🎯 Aim

The goal is to **Utilize customer behavioral and financial data to build a predictive model, enhancing accuracy and efficiency in risk assessment.**

**Business Goal:**  
- Reduce loan defaults by flagging high-risk applicants.
- Improve approval speed with automated scoring.

**Key Success Metrics:**  
- **Precision-Recall AUC** for imbalanced dataset performance.
- **Recall at threshold** to capture most defaulters.
- **False Positive Rate** minimization for better customer experience.

---

## 📊 Dataset

- **Source:** [Specify source or anonymized dataset]
- **Size:** ~XX,XXX rows × XX columns
- **Key Features:** Loan Amount, Term Days, Interest Rate, Repayment Ratio, Credit History, etc.
- **Challenges:**
  - Missing values
  - Outliers in loan amounts
  - Highly imbalanced target variable

---

## 🛠️ Approach  

**Exploratory Data Analysis (EDA)**  
- Analyzed distributions, trends, and correlations in borrower and loan data  
- Identified features with the strongest relationship to default risk  
- Flagged data quality concerns such as missing values and extreme outliers  

**Data Preprocessing**  
- Imputed missing values using strategies tailored for numeric and categorical variables  
- Detected and treated outliers to reduce skewness and improve model stability  
- Encoded categorical variables into numeric formats for model compatibility  
- Applied feature scaling to ensure consistent ranges across variables

 **Feature Selection**  
- Used **Variance Threshold** to remove low-variance features that provide little predictive value  

**Class Imbalance Handling**   
- Applied **SMOTE** (Synthetic Minority Oversampling Technique) to create realistic synthetic minority samples 


**Feature Engineering**
- late_payment_rate
- age
- repayment_efficiency

---

**Visualizations**
- Confusion Matrix
- ROC Curve


---

## 🚀 Deployment

**Local Usage**
```python
import joblib

model = joblib.load("catboost_model.pkl")
prediction = model.predict_proba(new_data)[:, 1]
