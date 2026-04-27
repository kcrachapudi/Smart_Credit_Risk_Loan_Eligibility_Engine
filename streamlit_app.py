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

# --- PHASE 3: THE CLEANING (Silver Layer) ---
st.divider()
st.header("Phase 3: The Silver Layer (Cleaned Data)")

if st.button("Execute Smart Imputation"):
    st.session_state.clean_df = clean_loan_data(st.session_state.raw_df)
    st.success("✨ Transformation Complete!")
    # UPDATED SYNTAX FOR 2026
    st.dataframe(st.session_state.clean_df.head(10), width='stretch')

# --- PHASE 4: FEATURE ENGINEERING (The Matrix) ---
st.divider()
st.header("Phase 4: Feature Engineering")
st.markdown("#### *Converting Categorical Data for the AI Model*")

if 'clean_df' in st.session_state:
    if st.button("Generate Feature Matrix"):
        from src.data_prep import encode_categories
        
        st.session_state.final_df = encode_categories(st.session_state.clean_df)
        
        st.success("🤖 Categorical text (Job/Reason) converted to numerical vectors.")
        # UPDATED SYNTAX FOR 2026
        st.dataframe(st.session_state.final_df.head(5), width='stretch')
        st.write(f"New Matrix Dimensions: {st.session_state.final_df.shape}")
else:
    st.info("👆 Please complete Phase 3 above first.")

# --- PHASE 5: THE SPLIT ---
st.divider()
st.header("Phase 5: The Train/Test Split")

if 'final_df' in st.session_state:
    if st.button("Generate Training & Testing Sets"):
        from src.data_prep import split_loan_data
        
        X_train, X_test, y_train, y_test = split_loan_data(st.session_state.final_df)
        
        # Save to state
        st.session_state.X_train = X_train
        st.session_state.y_train = y_train
        
        col1, col2 = st.columns(2)
        col1.metric("Training Records", len(X_train))
        col2.metric("Testing Records", len(X_test))
        st.success("🎯 Data split! We are ready to train the model.")

# --- PHASE 6: MACHINE LEARNING TRAINING ---
st.divider()
st.header("Phase 6: Training the AI Model")

# We need the training data from Session State
if 'X_train' in st.session_state:
    if st.button("🚀 Train Random Forest Model"):
        from src.data_prep import train_loan_model
        
        with st.spinner("Teaching the model to identify risk patterns..."):
            model = train_loan_model(st.session_state.X_train, st.session_state.y_train)
            st.session_state.loan_model = model
            
        st.success("✅ Model Trained Successfully!")
        st.balloons() # A little celebration for your first AI model!
        
        # Display Feature Importance (Which features did the AI find most useful?)
        st.subheader("What the AI Learned (Feature Importance)")
        importances = pd.Series(model.feature_importances_, index=st.session_state.X_train.columns).sort_values(ascending=False)
        st.bar_chart(importances)
else:
    st.info("👆 Please split the data in Phase 5 first.")
