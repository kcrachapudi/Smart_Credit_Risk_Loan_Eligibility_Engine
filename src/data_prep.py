# src/data_prep.py
import pandas as pd

def get_missing_stats(df):
    """Returns only columns that have missing values."""
    missing = df.isnull().sum()
    return missing[missing > 0]

def clean_loan_data(df):
    """
    Business Logic: The 'Silver Layer' cleaning process.
    """
    df_clean = df.copy()
    
    # Financial Logic: If derogatory reports are NaN, assume 0.
    zero_fill = ['DEROG', 'DELINQ', 'NINQ']
    for col in zero_fill:
        df_clean[col] = df_clean[col].fillna(0)
        
    # Financial Logic: For continuous debt values, use Median to avoid outlier bias.
    median_fill = ['MORTDUE', 'VALUE', 'YOJ', 'CLAGE', 'CLNO', 'DEBTINC']
    for col in median_fill:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
    # Categorical Logic: Mark missing as 'Unknown'
    df_clean['REASON'] = df_clean['REASON'].fillna('Unknown')
    df_clean['JOB'] = df_clean['JOB'].fillna('Other')
    
    return df_clean

def get_correlations(df):
    """
    Calculates how much each feature relates to the 'BAD' (Default) column.
    We exclude non-numeric columns for this calculation.
    """
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    # Get correlation with the target 'BAD' and sort it
    corr = numeric_df.corr()['BAD'].sort_values(ascending=False)
    return corr
