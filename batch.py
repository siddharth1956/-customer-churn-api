import pandas as pd
import requests
import argparse
import logging
import os

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(filename='logs/batch_log.txt', level=logging.INFO)

def main(input_file):
    df = pd.read_csv(input_file)
    results = []
    failures = 0
    total_prob = 0

    for _, row in df.iterrows():
        customer = row.drop("customerID", errors='ignore').to_dict()

        try:
            response = requests.post(
                "http://localhost:8000/predict",
                json={"customer": customer}
            )

            if response.status_code == 200:
                output = response.json()
                results.append({**customer, **output})
                total_prob += output["churn_probability"]
            else:
                failures += 1

        except Exception as e:
            logging.error(f"Request failed: {e}")
            failures += 1

    # Save predictions
    pd.DataFrame(results).to_csv("scored_customers.csv", index=False)

    # Log summary
    total = len(df)
    success = len(results)
    avg_prob = total_prob / success if success else 0
    logging.info(f"Total: {total}, Success: {success}, Failures: {failures}, Avg Prob: {avg_prob:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    main(args.input)