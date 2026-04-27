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


from sklearn.model_selection import train_test_split

def split_loan_data(df):
    """
    Splits data into Features (X) and Target (y), 
    then into Training and Testing sets.
    """
    # X = everything except the target 'BAD'
    X = df.drop('BAD', axis=1)
    # y = only the 'BAD' column
    y = df['BAD']
    
    # Stratify ensures the 80/20 'paid/default' ratio is preserved in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test


# src/data_prep.py (Add this if it's missing!)

def encode_categories(df):
    """
    Converts 'REASON' and 'JOB' into numerical columns 
    so the Machine Learning model can process them.
    """
    import pandas as pd
    # This creates new columns like JOB_Office, JOB_Sales, etc.
    df_encoded = pd.get_dummies(df, columns=['REASON', 'JOB'], drop_first=True)
    return df_encoded

from sklearn.ensemble import RandomForestClassifier

def train_loan_model(X_train, y_train):
    """
    Trains a Random Forest model to predict loan default.
    """
    # random_state=42 ensures you get the same result every time you run it
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    return model
