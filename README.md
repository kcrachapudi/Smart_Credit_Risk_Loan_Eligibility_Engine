# Smart-Credit_Risk-Loan-Eligibility-Engine
Smart Credit Risk Loan Eligibility Engine

## Objective
To build a machine learning pipeline that predicts the probability of loan default (Target: `BAD`) based on home equity loan applications.

## Data Dictionary (The Features)

| Feature | Meaning | Why it matters |
| :--- | :--- | :--- |
| **BAD** | 1 = Default, 0 = Repaid | **Our Target variable.** |
| **LOAN** | Amount of loan request | Larger loans often carry higher risk. |
| **MORTDUE** | Amount due on existing mortgage | Shows existing debt burden. |
| **VALUE** | Value of current property | Collateral for the loan. |
| **REASON** | DebtCon (Debt Consolidation) or HomeImp (Home Improvement) | Purpose of the loan affects risk profile. |
| **JOB** | Occupational categories | Stability of income source. |
| **YOJ** | Years at present job | Work stability. |
| **DEROG** | Number of major derogatory reports | Past credit "red flags." |
| **DELINQ** | Number of delinquent credit lines | Recent payment failures. |
| **CLAGE** | Age of oldest credit line (months) | Long credit history = lower risk. |
| **NINQ** | Number of recent credit inquiries | Too many inquiries can signal financial distress. |
| **CLNO** | Number of credit lines | Total credit footprint. |
| **DEBTINC** | Debt-to-income ratio | **The "Smoking Gun" for defaults.** |
