# 🧠 Customer Churn Prediction API

This project predicts whether a customer will churn (leave) using a machine learning model. It includes:

- A Flask API for real-time predictions
- A batch prediction script
- Trained model and preprocessing pipeline

---

## 📁 Project Structure
customer-churn-api/
│
├── app/
│   ├── init.py
│   ├── main.py              # Flask API
│   ├── model.pkl            # Trained model
│   ├── transformer.pkl      # Preprocessing pipeline
│   └── utils.py             # Helper functions
│
├── batch.py                 # Batch prediction script
├── train_model.py           # Model training script
├── sample_input.json        # Sample input for API
├── all_customers.csv        # Dataset
├── gold_churn_data.csv      # Ground truth for validation
├── scored_customers.csv     # Output from batch scoring
├── test_data/               # Test files
├── logs/                    # Batch logs
├── requirements.txt
└── README.
---

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
