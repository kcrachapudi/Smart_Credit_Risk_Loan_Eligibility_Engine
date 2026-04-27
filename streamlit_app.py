# streamlit_app.py
import streamlit as st
import pandas as pd
import os
from src.data_prep import get_missing_stats, clean_loan_data

# --- PAGE CONFIG ---
st.set_page_config(page_title="FinTech: Loan Risk Engine", layout="wide")

# --- TITLES & HEADINGS ---
st.title("🏦 Smart Credit Risk & Loan Eligibility Engine")
st.markdown("#### *The Story of a Loan: From Raw Data to Risk Prediction*")
st.divider()

# --- PHASE 1: AUTOMATIC INGESTION ---
st.header("Phase 1: Data Ingestion (The Bronze Layer)")

# Logic to load the default file
DATA_PATH = "data/hmeq.csv"

if os.path.exists(DATA_PATH):
    if 'raw_df' not in st.session_state:
        st.session_state.raw_df = pd.read_csv(DATA_PATH)
    
    st.success(f"✅ Automatically loaded: `{DATA_PATH}`")
    
    with st.expander("🔍 View Raw Applicant Records (First 10 rows)"):
        st.dataframe(st.session_state.raw_df.head(10), use_container_width=True)
else:
    st.error(f"❌ File not found at `{DATA_PATH}`. Please ensure the hmeq.csv is in your data folder.")
    st.stop() # Prevents the rest of the app from running until file is found

# --- PHASE 2: HEALTH CHECK & DIAGNOSTICS ---
st.divider()
st.header("Phase 2: Health Check & Diagnostics")

if st.button("Run Diagnostic Check"):
    missing_data = get_missing_stats(st.session_state.raw_df)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Missing Fields")
        st.dataframe(missing_data[missing_data > 0], use_container_width=True)
    with col2:
        st.subheader("The 'Target' Balance")
        # Visualizing the 80/20 imbalance we discussed
        target_counts = st.session_state.raw_df['BAD'].value_counts()
        st.bar_chart(target_counts)
        st.info("Goal: Predict the '1's (Defaults) without being fooled by the '0's (Paid).")

# --- PHASE 3: THE CLEANING ---
st.divider()
st.header("Phase 3: The Silver Layer (Cleaned Data)")

if st.button("Execute Smart Imputation"):
    st.session_state.clean_df = clean_loan_data(st.session_state.raw_df)
    
    st.success("✨ Transformation Complete!")
    st.dataframe(st.session_state.clean_df.head(10), use_container_width=True)
    
    # Quick Check: Are there any NaNs left?
    final_nans = st.session_state.clean_df.isnull().sum().sum()
    st.write(f"**Total Missing Values Remaining:** {final_nans}")
