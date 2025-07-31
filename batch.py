import argparse
import pandas as pd
import json
import requests
import logging
import os
from datetime import datetime

# Set up logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(filename="logs/batch_log.txt", level=logging.INFO)

def run_batch(input_file):
    df = pd.read_csv(input_file)

    # Drop "Unnamed: 0" if present
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    total = len(df)
    failures = 0
    probs = []

    results = []

    for i, row in df.iterrows():
        customer_json = json.dumps({"customer": row.to_dict()})
        try:
            response = requests.post(
                "http://localhost:8000/predict",
                data=customer_json,
                headers={"Content-Type": "application/json"}
            )
            if response.status_code == 200:
                output = response.json()
                row_result = row.to_dict()
                row_result.update(output)
                results.append(row_result)
                probs.append(output["churn_probability"])
            else:
                failures += 1
        except Exception as e:
            failures += 1
            continue

    pd.DataFrame(results).to_csv("scored_customers.csv", index=False)

    avg_prob = sum(probs) / len(probs) if probs else 0.0

    logging.info(f"Timestamp: {datetime.now()}")
    logging.info(f"Total requests: {total}")
    logging.info(f"Failures: {failures}")
    logging.info(f"Average churn probability: {avg_prob:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    args = parser.parse_args()
    run_batch(args.input)
