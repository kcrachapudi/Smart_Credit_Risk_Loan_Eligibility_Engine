# 🏦 Smart Credit Risk & Loan Eligibility Engine
### End-to-End FinTech Pipeline: From Raw Banking Data to AI-Driven Decisions

## 🎯 Project Overview
Financial institutions lose billions annually due to loan defaults. This project simulates a professional FinTech environment to build a robust **Credit Risk Assessment Tool**. 

Using the **HMEQ (Home Equity)** dataset, I built a modular pipeline that ingests raw banking data, cleans it using domain-specific financial logic, and trains a **Random Forest Classifier** to predict the probability of default in real-time.

## 🏗️ Technical Architecture
The project follows a **Medallion Architecture** to ensure data integrity:
- **Bronze Layer (Raw):** Direct ingestion of messy CSV data with over 20% missing values.
- **Silver Layer (Cleaned):** Implemented smart imputation (Median-fill for debt ratios, Zero-fill for credit events) to preserve 100% of the applicant pool.
- **Gold Layer (Feature Matrix):** One-Hot Encoding of categorical variables (Job, Reason) for model compatibility.

## 🛠️ Tech Stack
- **OS:** Ubuntu (LinData VM)
- **Language:** Python 3.12
- **UI Framework:** Streamlit (2026 Interactive "Story" Mode)
- **Data Science:** Pandas, NumPy, Scikit-learn
- **ML Model:** Random Forest Classifier (with Class Weight Balancing)
- **Version Control:** Git & GitHub

## 📈 Key Insights & Features
- **Risk Drivers:** Through correlation and feature importance analysis, the model identified **DELINQ** (past delinquencies) and **DEBTINC** (Debt-to-Income ratio) as the primary indicators of risk.
- **Real-Time Eligibility Checker:** An interactive interface allowing loan officers to input custom applicant data and receive a risk probability score and an automated "Approve/Reject" decision.
- **Class Imbalance Handling:** Utilized Stratified Splitting to ensure the model learned effectively from a dataset where only 20% of applicants defaulted.

## 🚀 How to Run
1. **Clone the Repo:** 
   `git clone https://github.com`
2. **Setup Virtual Env:** 
   `python3 -m venv venv && source venv/bin/activate`
3. **Install Dependencies:** 
   `pip install -r requirements.txt`
4. **Launch the Story:** 
   `streamlit run streamlit_app.py`

---
*Developed as part of a FinTech Portfolio focused on Data Engineering and Machine Learning.*
