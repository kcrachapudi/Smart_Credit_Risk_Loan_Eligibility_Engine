# streamlit_app.py
import streamlit as st
import pandas as pd
import os

# Import our engine from the src folder
from src.data_prep import (
    get_missing_stats, 
    clean_loan_data, 
    get_correlations, 
    encode_categories, 
    split_loan_data,
    train_loan_model,
    predict_eligibility
)

# --- PAGE CONFIG ---
st.set_page_config(page_title="FinTech: Loan Risk Engine", layout="wide")

# --- TITLES & HEADINGS ---
st.title("🏦 Smart Credit Risk & Loan Eligibility Engine")
st.markdown("#### *The Story of a Loan: From Raw Data to Risk Prediction*")
st.divider()

# --- PHASE 1: DATA INGESTION (Bronze Layer) ---
st.header("Phase 1: Data Ingestion (The Bronze Layer)")
DATA_PATH = "data/hmeq.csv"

if os.path.exists(DATA_PATH):
    if 'raw_df' not in st.session_state:
        st.session_state.raw_df = pd.read_csv(DATA_PATH)
    
    st.success(f"✅ Automatically loaded: `{DATA_PATH}`")
    with st.expander("🔍 View Raw Applicant Records"):
        st.dataframe(st.session_state.raw_df.head(10), width='stretch')
else:
    st.error(f"❌ File not found at `{DATA_PATH}`. Please check your data folder.")
    st.stop()

# --- PHASE 2: HEALTH CHECK & DIAGNOSTICS ---
st.divider()
st.header("Phase 2: Health Check & Diagnostics")

if st.button("Run Diagnostic Check"):
    st.session_state.run_diag = True

if st.session_state.get('run_diag'):
    missing_data = get_missing_stats(st.session_state.raw_df)
    
    st.subheader("1. Missing Fields")
    st.dataframe(missing_data[missing_data > 0], width='stretch')
    
    st.write("") 
    st.subheader("2. The 'Target' Balance")
    target_counts = st.session_state.raw_df['BAD'].value_counts()
    st.bar_chart(target_counts)
    st.info("Imbalance: ~20% Defaults (1) vs ~80% Paid (0).")

# --- PHASE 3: THE CLEANING (Silver Layer) ---
st.divider()
st.header("Phase 3: The Silver Layer (Cleaned Data)")

if st.button("Execute Smart Imputation"):
    st.session_state.clean_df = clean_loan_data(st.session_state.raw_df)

if 'clean_df' in st.session_state:
    st.success("✨ Transformation Complete: Missing values filled logically.")
    st.dataframe(st.session_state.clean_df.head(10), width='stretch')

# --- PHASE 4: FEATURE ENGINEERING ---
st.divider()
st.header("Phase 4: Feature Engineering")

if 'clean_df' in st.session_state:
    if st.button("Generate Feature Matrix"):
        st.session_state.final_df = encode_categories(st.session_state.clean_df)
    
    if 'final_df' in st.session_state:
        st.success("🤖 Categorical variables encoded into numerical vectors.")
        st.dataframe(st.session_state.final_df.head(5), width='stretch')
        
        # Risk Drivers Analysis (Correlation)
        st.subheader("Risk Driver Analysis")
        correlations = get_correlations(st.session_state.final_df)
        st.bar_chart(correlations.drop('BAD'))
else:
    st.info("👆 Please clean the data in Phase 3 first.")

# --- PHASE 5: THE TRAIN/TEST SPLIT ---
st.divider()
st.header("Phase 5: The Train/Test Split")

if 'final_df' in st.session_state:
    if st.button("Generate Training & Testing Sets"):
        X_train, X_test, y_train, y_test = split_loan_data(st.session_state.final_df)
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test

    if 'X_train' in st.session_state:
        col1, col2 = st.columns(2)
        col1.metric("Training Records", len(st.session_state.X_train))
        col2.metric("Testing Records", len(st.session_state.X_test))
        st.success("🎯 Data split and stratified! Ready for the AI.")

# --- PHASE 6: MACHINE LEARNING TRAINING ---
st.divider()
st.header("Phase 6: Training the AI Model")

if 'X_train' in st.session_state:
    if st.button("🚀 Train Random Forest Model"):
        with st.spinner("Teaching the model to identify risk patterns..."):
            model = train_loan_model(st.session_state.X_train, st.session_state.y_train)
            st.session_state.loan_model = model
            st.balloons()

    if 'loan_model' in st.session_state:
        st.success("✅ Model Trained Successfully!")
        st.subheader("What the AI Learned (Feature Importance)")
        model = st.session_state.loan_model
        importances = pd.Series(
            model.feature_importances_, 
            index=st.session_state.X_train.columns
        ).sort_values(ascending=False)
        st.bar_chart(importances)
else:
    st.info("👆 Please split the data in Phase 5 first.")


# --- PHASE 7: LIVE ELIGIBILITY CHECKER ---
st.divider()
st.header("Phase 7: Real-Time Loan Eligibility Checker")
st.markdown("#### *Test the Model with Custom Applicant Data*")

if 'loan_model' in st.session_state:
    with st.container():
        col1, col2, col3 = st.columns(3)
        
        # User Inputs
        loan_amt = col1.number_input("Loan Amount ($)", value=20000)
        mortdue = col2.number_input("Mortgage Due ($)", value=50000)
        value = col3.number_input("Property Value ($)", value=85000)
        
        col4, col5, col6 = st.columns(3)
        debtinc = col4.slider("Debt-to-Income Ratio (%)", 0, 100, 25)
        clage = col5.slider("Credit History (Months)", 0, 600, 150)
        # Fix: Added the options [0, 1, 2, 3, 4, 5] for the dropdown
        delinq = col6.selectbox("Recent Delinquencies", options=[0, 1, 2, 3, 4, 5])

    if st.button("🔍 Check Eligibility"):
        
        user_input = pd.DataFrame({
            'LOAN': [loan_amt], 'MORTDUE': [mortdue], 'VALUE': [value],
            'DEBTINC': [debtinc], 'CLAGE': [clage], 'DELINQ': [delinq]
        })
        
        # Get the prediction array
        risk_probs = predict_eligibility(
            st.session_state.loan_model, 
            user_input, 
            st.session_state.X_train.columns
        )
        
        # SAFE UNPACKING:
        # Scikit-learn usually returns [[p0, p1]]. 
        # We flatten it to [p0, p1] and take the second value.
        default_prob = risk_probs.flatten()[1] 
        
        st.subheader(f"Risk Probability: {default_prob:.1%}")
        
        if default_prob < 0.3:
            st.success("✅ LOAN APPROVED: Low risk of default.")
        elif default_prob < 0.6:
            st.warning("⚠️ MANUAL REVIEW REQUIRED: Moderate risk level.")
        else:
            st.error("❌ LOAN REJECTED: High risk of default detected.")
