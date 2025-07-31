# 🧠 Customer Churn Prediction API

This project predicts whether a customer will churn (leave) using a machine learning model. It includes:

- A Flask API for real-time predictions
- A batch prediction script
- Trained model and preprocessing pipeline

---

## 📁 Project Structure
<details>
<summary>customer-churn-api/
├── app/
│   ├── init.py
│   ├── main.py
│   ├── model.pkl
│   ├── transformer.pkl
│   └── utils.py
├── batch.py
├── train_model.py
├── all_customers.csv
├── gold_churn_data.csv
├── scored_customers.csv
├── sample_input.json
├── test_data/
│   ├── all_customers.csv
│   └── sample_input.json
├── logs/
├── requirements.txt
└── README.md
</summary>
</details>


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
