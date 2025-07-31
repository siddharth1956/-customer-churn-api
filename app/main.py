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

        # If the JSON has a 'customer' key, use it. Otherwise use the full data.
        customer_data = data.get("customer", data)
        df = pd.DataFrame([customer_data])

        # Drop 'Unnamed: 0' if it exists
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])

        # Transform and predict
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
