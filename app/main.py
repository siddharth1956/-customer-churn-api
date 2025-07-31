from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load("app/model.pkl")
transformer = joblib.load("app/transformer.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data["customer"]])
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