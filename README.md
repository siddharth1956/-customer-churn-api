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


🧪 Auto-Grading Support

This repo is compatible with auto-graders and CI/CD:
	•	✅ Cloneable
	•	✅ requirements.txt installs all dependencies
	•	✅ python -m app.main launches the API
	•	✅ curl or POST requests to /predict work
	•	✅ batch.py scores customer data
	•	✅ Logs and output (scored_customers.csv) are generated

⸻

📦 Setup Instructions

1. Clone the Repository

git clone https://github.com/siddharth1956/-customer-churn-api.git
cd -customer-churn-api

2. Create & Activate Virtual Environment

python3 -m venv venv
source venv/bin/activate  # Mac/Linux
# OR
venv\Scripts\activate    # Windows

3. Install Dependencies

pip install -r requirements.txt


⸻

🚀 Running the Real-Time API

python -m app.main

Access it at: http://localhost:8000

Test with:

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @test_data/sample_input.json


⸻

📊 Batch Prediction

python batch.py --input test_data/all_customers.csv

Output will be saved to scored_customers.csv

⸻

🔧 Maintenance Plan
	•	📅 Retrain every 3 months using updated customer data
	•	📉 Monitor drift by comparing model prediction distributions
	•	🪲 Log API failures and batch errors to logs/
	•	📌 Version models by naming them with timestamps

⸻

📬 Contact

Author: Siddharth Shetty
GitHub: siddharth1956
