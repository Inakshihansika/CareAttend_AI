# 🏥 CareAttend AI

### Medical Appointment No-Show Prediction using Machine Learning

> **CareAttend AI** is an interactive machine learning application that predicts the likelihood of a patient missing a scheduled medical appointment. It provides risk scoring, analytics, model comparison, and batch prediction through a user-friendly Streamlit dashboard.

---

## 🚀 Live Demo

### 🌐 Try CareAttend AI

**[Open the Live Streamlit App](https://careattendai-qqj2and3krhtpzgjbnkjcu.streamlit.app/)**

Explore the application and test appointment risk predictions directly in your browser.

---

## 📌 Project Overview

Missed medical appointments can lead to unused appointment slots, inefficient resource utilization, and scheduling difficulties.

**CareAttend AI** uses machine learning to identify appointments that may have a higher probability of becoming a no-show.

The system analyzes patient and appointment-related information and generates:

* 🎯 No-show prediction
* 📊 0–100 risk score
* 🟢 Low risk
* 🟠 Medium risk
* 🔴 High risk
* 💡 Recommended operational actions
* 📈 Interactive analytics
* 🤖 Machine learning model comparison
* 📂 Batch CSV prediction

The project demonstrates an end-to-end **Data Science + Machine Learning + Streamlit deployment workflow**.

---

## 🎯 Objectives

* Predict whether a patient is likely to miss an appointment.
* Identify appointments with higher no-show risk.
* Compare multiple classification algorithms.
* Evaluate models using appropriate classification metrics.
* Provide an interactive interface for individual predictions.
* Support batch prediction using CSV files.
* Present insights through an easy-to-use healthcare dashboard.

---

## 🧠 Machine Learning Models

The project compares three classification algorithms:

| Model               | Purpose                            |
| ------------------- | ---------------------------------- |
| Logistic Regression | Baseline classification model      |
| Random Forest       | Ensemble tree-based model          |
| Gradient Boosting   | Sequential ensemble learning model |

The best-performing model is automatically selected based on **F1 Score**.

---

## 📊 Model Evaluation

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Feature Importance

### Why F1 Score?

Medical appointment no-show prediction is a classification problem where both false positives and false negatives matter. Therefore, **F1 Score** is used as the primary criterion for selecting the best model.

---

## 🔍 Features Used

The model uses the following features:

* Age
* Gender
* Scholarship
* Hypertension
* Diabetes
* Alcoholism
* Handicap
* SMS Received
* Waiting Days
* Appointment Weekday
* Appointment Month

### Feature Engineering

Appointment dates are transformed into useful machine learning features:

```text
Scheduled Date
       +
Appointment Date
       ↓
Waiting Days
Appointment Weekday
Appointment Month
```

---

## 🖥️ Application Features

### 🏠 Dashboard

Provides an executive overview containing:

* Total appointments
* Attended appointments
* No-show appointments
* Overall no-show rate
* Attendance distribution
* Project objective

### 👤 Patient Risk Prediction

Enter individual patient and appointment information to receive:

```text
Prediction
     ↓
Probability
     ↓
Risk Score (0–100)
     ↓
Risk Level
     ↓
Recommended Action
```

Example risk categories:

| Risk Score | Risk Level |
| ---------: | ---------- |
|       0–39 | 🟢 LOW     |
|      40–69 | 🟠 MEDIUM  |
|     70–100 | 🔴 HIGH    |

### 📂 Batch Prediction

Upload a CSV containing multiple appointment records and generate:

* Prediction
* Risk Score
* Risk Level

The results can also be downloaded as a CSV file.

### 📊 Analytics

Explore appointment patterns using interactive filters and visualizations.

Current analysis includes:

* No-show rate by gender
* No-show rate based on SMS reminders
* Filtered appointment analysis

### 🧠 Model Lab

Compare:

* Logistic Regression
* Random Forest
* Gradient Boosting

Also includes:

* Accuracy comparison
* Precision comparison
* Recall comparison
* F1 Score comparison
* Confusion Matrix
* Feature Importance

### 🔎 Dataset Explorer

Explore:

* Dataset records
* Number of rows
* Number of columns
* Missing values
* Data types
* Unique values

### ℹ️ About

Provides information about the project, technologies, models, and developer.

---

## 🛠️ Technology Stack

### Programming

* Python

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-learn

### Visualization

* Matplotlib
* Seaborn

### Web Application

* Streamlit

### Development

* VS Code
* Git
* GitHub

### Deployment

* Streamlit Community Cloud

---

## 📁 Project Structure

```text
CareAttend_AI/
│
├── app.py
├── model.py
├── medical_appointments.csv
├── requirements.txt
└── README.md
```

### File Description

| File                       | Description                                             |
| -------------------------- | ------------------------------------------------------- |
| `app.py`                   | Streamlit application and dashboard                     |
| `model.py`                 | Data preprocessing, model training and prediction logic |
| `medical_appointments.csv` | Medical appointment dataset                             |
| `requirements.txt`         | Python dependencies                                     |
| `README.md`                | Project documentation                                   |

---

## ⚙️ Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Inakshihansika/CareAttend_AI.git
```

### 2. Open the project

```bash
cd CareAttend_AI
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📦 Requirements

The project uses:

```text
streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
```

---

## 🔄 Machine Learning Workflow

```text
Medical Appointment Dataset
            ↓
      Data Cleaning
            ↓
   Data Preprocessing
            ↓
    Feature Engineering
            ↓
      Train/Test Split
            ↓
   ┌─────────────────────┐
   │  Logistic Regression│
   │  Random Forest      │
   │  Gradient Boosting  │
   └─────────────────────┘
            ↓
      Model Evaluation
            ↓
       Best Model
            ↓
     Risk Prediction
            ↓
   Streamlit Dashboard
```

---

## 💡 Example Use Case

A clinic can use the system to identify appointments that may have a higher probability of no-show.

For example:

```text
Patient Information
        ↓
CareAttend AI
        ↓
Risk Score: 78/100
        ↓
Risk Level: HIGH
        ↓
Recommended Action:
Additional SMS reminder /
Confirmation call
```

This can help prioritize reminder efforts rather than treating every appointment identically.

---

## 🔮 Future Improvements

Possible future enhancements include:

* 📱 SMS/WhatsApp reminder integration
* 📅 Appointment scheduling optimization
* 📈 Advanced time-series analysis
* 🤖 XGBoost and other advanced models
* 🔍 Explainable AI using SHAP
* 🏥 Hospital/clinic dashboard integration
* 👤 Patient history-based features
* ⚡ Real-time prediction API
* 🔐 Enhanced healthcare data privacy controls

---

## ⚠️ Disclaimer

CareAttend AI is an **educational and demonstration project**.

The predictions are generated by machine learning models trained on the available dataset and **are not clinically validated**. The application should not be used as a substitute for professional medical judgment or real-world clinical decision-making.

---

## 👩‍💻 Developer

### Inakshi Hansika S S

**B.Sc. Data Science**

Data Science | Machine Learning | Analytics | Python | Streamlit

---

## 🌐 Project Links

**🚀 Live Application:**
https://careattendai-qqj2and3krhtpzgjbnkjcu.streamlit.app/

**💻 GitHub Repository:**
https://github.com/Inakshihansika/CareAttend_AI

---

### ⭐ If you found this project interesting, consider giving the repository a star!
