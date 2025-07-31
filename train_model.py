import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

# Load the training data
df = pd.read_csv('gold_churn_data.csv')

# Drop unnecessary columns
df = df.drop(columns=['customerID'], errors='ignore')

# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Drop rows with missing target
df = df.dropna(subset=['Churn'])

# Map target column
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Features and target
X = df.drop(columns=['Churn'])
y = df['Churn']

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include='object').columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ('num', SimpleImputer(strategy='mean'), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
])

# Final pipeline with model
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Train the model
pipeline.fit(X, y)

# Save components
joblib.dump(pipeline.named_steps['preprocessor'], 'app/transformer.pkl')
joblib.dump(pipeline.named_steps['classifier'], 'app/model.pkl')
