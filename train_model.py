import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# Load your training data
df = pd.read_csv("gold_churn_data.csv")

X = df.drop(columns=["Churn", "customerID"])
y = df["Churn"].map({"Yes": 1, "No": 0})
X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")

# Feature types
cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Preprocessing
preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="mean"), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
])

# Train model
X_cleaned = preprocessor.fit_transform(X)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_cleaned, y)

# Save both
joblib.dump(model, "app/model.pkl")
joblib.dump(preprocessor, "app/transformer.pkl")