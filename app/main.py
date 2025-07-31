from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and transformer
model = joblib.load("app/model.pkl")
transformer = joblib.load("app/transformer.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Convert incoming JSON to DataFrame
        df = pd.DataFrame([data["customer"]])

        # Fix: Add dummy 'Unnamed: 0' column if required
        if 'Unnamed: 0' not in df.columns:
            df['Unnamed: 0'] = 0  # Add a default dummy index value

        # Transform the input and make prediction
        X = transformer.transform(df)
        prob = model.predict_proba(X)[0][1]
        prediction = "Yes" if prob > 0.5 else "No"

        return jsonify({
            "churn_probability": round(prob, 2),
            "churn_prediction": prediction
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000)
