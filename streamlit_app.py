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

# --- PHASE 2: HEALTH CHECK & DIAGNOSTICS (Vertical Layout) ---
st.divider()
st.header("Phase 2: Health Check & Diagnostics")

if st.button("Run Diagnostic Check"):
    missing_data = get_missing_stats(st.session_state.raw_df)
    
    # Grid first
    st.subheader("1. Missing Fields")
    st.dataframe(missing_data[missing_data > 0], use_container_width=True)
    
    # Chart below with some extra spacing
    st.write("") 
    st.subheader("2. The 'Target' Balance")
    target_counts = st.session_state.raw_df['BAD'].value_counts()
    st.bar_chart(target_counts)
    
    st.info("The imbalance is clear: We have many more 'Paid' (0) examples than 'Default' (1) examples.")

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


# --- PHASE 4: EXPLORATORY DATA ANALYSIS (EDA) ---
st.divider()
st.header("Phase 4: Identifying Risk Drivers")

if 'clean_df' in st.session_state:
    if st.button("Analyze Risk Correlations"):
        from src.data_prep import get_correlations
        
        correlations = get_correlations(st.session_state.clean_df)
        
        st.subheader("What predicts a Default?")
        # We drop 'BAD' from the chart so it doesn't correlate with itself (1.0)
        st.bar_chart(correlations.drop('BAD'))
        
        st.info("""
        **How to read this:**
        - **Positive bars (up):** As these increase (like DELINQ or DEROG), the risk of default goes UP.
        - **Negative bars (down):** As these increase (like CLAGE or DEBTINC), the risk of default actually goes DOWN (or vice versa).
        """)
else:
    st.info("👆 Please clean the data in Phase 3 first to enable analysis.")


