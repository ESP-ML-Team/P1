


# Data Cleaning
column ID is excel row minus 2
see column_metadata.ods and https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf


Identify relevant predictors 
Demographics: (age, sex, race, education, income)
Behavioral Factors: (Smoking status, physical activity, diet)
Medical History: (Hypertension, cholesterol levels, diabetes)

Missing values
Outliers
Duplicates
Recoding special codes
Core vs. Optional questions by State
Derived Variables

Create binary variables
Categorical/Continuous variables
•	Nominal variables (like “color”, “country”) should be one-hot encoded.
•	Ordinal variables (like “size”, “ranking”) can be label-encoded or ordinal-encoded.

Sampling weights (LLCPWT)
STRWT, PSU


# Model Training

## Class imbalance
y_train:
-1 299,160
1 28,975

Over/under sampling, SMOTE, AUC-ROC, Precission/Recall, F1

## Other models
Random forest
KSVM, RDA