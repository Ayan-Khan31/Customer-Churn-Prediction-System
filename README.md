# Customer Churn Prediction System

## Overview

The Customer Churn Prediction System is an end-to-end machine learning application designed to predict whether a customer is likely to churn based on customer behavior, contract information, payment methods, and service usage patterns.

This project combines:

* Machine Learning
* Business Analytics
* Interactive Dashboards
* Real-Time Prediction Systems

The application was built using Python, Scikit-learn, Streamlit, and Plotly to simulate a real-world AI-powered business intelligence platform.

---

## Features

✅ Customer churn prediction using Machine Learning
✅ Random Forest classification model
✅ Real-time churn probability analysis
✅ KPI dashboard cards
✅ Risk classification system
✅ Business recommendation engine
✅ Interactive analytics visualizations
✅ Streamlit-based web application
✅ Business Insights dashboard
✅ Feature importance analysis

---

## Tech Stack

### Programming & Data

* Python
* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* Random Forest Classifier

### Visualization

* Plotly

### Web Application

* Streamlit

---

## Machine Learning Workflow

### 1. Data Preprocessing

* Handled categorical encoding using One-Hot Encoding
* Feature engineering and preprocessing
* Train-test split for evaluation

### 2. Model Training

* Random Forest Classifier used for prediction
* Feature importance analysis performed
* Model evaluation completed using accuracy metrics

### 3. Deployment

* Trained model saved using Pickle
* Interactive Streamlit application built for real-time predictions

---

## Model Performance

| Metric                 | Value  |
| ---------------------- | ------ |
| Random Forest Accuracy | 79.84% |

---

## Key Business Insights

* Customers with low tenure are more likely to churn
* Month-to-month contracts increase churn probability
* High monthly charges correlate with customer loss
* Electronic check payment users show higher churn risk
* Long-term contracts improve retention rates

---

## Dashboard Features

### Prediction Dashboard

* Customer input interface
* Real-time churn prediction
* Probability score
* KPI metrics
* Risk visualization
* Business action recommendations

### Business Insights Dashboard

* Feature importance chart
* Retention strategy recommendations
* Churn driver analysis
* Business KPI overview

---

## Dashboard Preview

### Prediction Dashboard

![Prediction Dashboard](screenshots/Prediction_Dashboard.png)

### Business Insights Dashboard

![Business Insights](screenshots/Business_Insights.png)

---

## Project Structure

```bash
Customer Churn Prediction/
│
├── app/
│   └── app.py
│
├── models/
│   ├── churn_model.pkl
│   └── feature_names.pkl
│
├── screenshots/
│   ├── prediction_dashboard.png
│   └── business_insights.png
│
├── notebooks/
│
├── requirements.txt
│
└── README.md
```

---

## Installation & Setup

### Clone Repository

```bash
git clone YOUR_GITHUB_REPO_LINK
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
python -m streamlit run app/app.py
```

---

## Future Improvements

* Batch CSV prediction uploads
* Cloud deployment
* User authentication
* Customer segmentation
* Deep learning implementation
* API integration
* Automated reporting system

---

## Business Value

This project demonstrates how machine learning can help businesses:

* Identify at-risk customers
* Improve retention strategies
* Reduce revenue loss
* Generate actionable customer insights
* Support data-driven decision making

---

## Author

Developed by Ayan as a Machine Learning & Business Analytics project focused on real-world AI deployment and predictive analytics systems.
