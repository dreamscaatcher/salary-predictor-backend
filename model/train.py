import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.metrics import mean_squared_error, r2_score
import joblib

import os

# Get absolute paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'ds_salaries.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'lin_regress.sav')

# Load data
salary_data = pd.read_csv(DATA_PATH)

# Keep only relevant columns
salary_data = salary_data[['experience_level', 'company_size', 'employment_type', 'job_title', 'salary_in_usd']]

# Use ordinal encoder for experience level
encoder = OrdinalEncoder(categories=[['EN', 'MI', 'SE', 'EX']])
salary_data['experience_level_encoded'] = encoder.fit_transform(salary_data[['experience_level']])

# Use ordinal encoder for company size
encoder = OrdinalEncoder(categories=[['S', 'M', 'L']])
salary_data['company_size_encoded'] = encoder.fit_transform(salary_data[['company_size']])

# Create dummy for part-time
salary_data['employment_type_PT'] = (salary_data['employment_type'] == 'PT').astype(int)

# Create dummies for specific job titles we want to predict for
job_titles = ['Data Engineer', 'Data Manager', 'Data Scientist', 'Machine Learning Engineer']
for title in job_titles:
    salary_data[f'job_title_{title.replace(" ", "_")}'] = (salary_data['job_title'] == title).astype(int)

# Select features for model
features = [
    'experience_level_encoded',
    'company_size_encoded',
    'employment_type_PT',
    'job_title_Data_Engineer',
    'job_title_Data_Manager',
    'job_title_Data_Scientist',
    'job_title_Machine_Learning_Engineer'
]

# Define features and target
X = salary_data[features]
y = salary_data['salary_in_usd']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=104, test_size=0.2, shuffle=True)

# Train model
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

# Make predictions
y_pred = regr.predict(X_test)

# Print model performance
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print("R2 score: %.2f" % r2_score(y_test, y_pred))

# Save model
joblib.dump(regr, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
