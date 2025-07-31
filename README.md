# 🧠 Customer Churn Prediction API

This project predicts whether a customer will churn (leave) using a machine learning model. It includes:

- A Flask API for real-time predictions
- A batch prediction script
- Trained model and preprocessing pipeline

---

## 📁 Project Structure

<details>
<summary>Click to expand</summary>

<br>

```plaintext
customer-churn-api/
├── app/
│   ├── __init__.py
│   ├── main.py              # Flask API
│   ├── model.pkl            # Trained model
│   ├── transformer.pkl      # Preprocessing pipeline
│   └── utils.py             # Helper functions
├── batch.py                 # Batch scoring script
├── train_model.py           # Model training script
├── all_customers.csv        # Input customer data
├── gold_churn_data.csv      # Ground truth labels
├── scored_customers.csv     # Output from batch scoring
├── sample_input.json        # Sample for real-time API
├── test_data/
│   ├── all_customers.csv
│   └── sample_input.json
├── logs/                    # Log files
├── requirements.txt         # Project dependencies
└── README.md
</details>
```

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/siddharth1956/-customer-churn-api.git
cd customer-churn-api
2. Install Dependencies
pip install -r requirements.txt
3. Start the Flask API
python -m app.main
API will run on http://localhost:8000/predict.
4. Run Batch Prediction
python batch.py --input test_data/all_customers.csv
Output will be saved in scored_customers.csv.
📦 Sample API Input
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 5,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  ...
}
🧠 Model
	•	Logistic Regression or Tree-based model
	•	Trained on Telco Customer Churn Dataset
	•	Preprocessed using Scikit-learn pipeline

⸻

📌 Author

Siddharth Shetty
