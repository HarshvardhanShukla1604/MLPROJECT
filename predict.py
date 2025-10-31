"""
Utility to load a trained artifact (saved by `app.py`) and run predictions on a CSV.

Produces a CSV with predictions added as a column named `prediction` (original class
labels) and writes to --out (default: predictions.csv).

Usage example:
	python predict.py --csv data/new_data.csv --model trained_model.pkl --out preds.csv
"""

import argparse
import os
import sys
import pandas as pd
import joblib


def load_artifact(path: str):
	if not os.path.isfile(path):
		print(f"Model artifact not found: {path}")
		sys.exit(1)
	return joblib.load(path)


def predict(csv_path: str, model_path: str, out_path: str):
	artifact = load_artifact(model_path)
	model = artifact.get("model")
	le = artifact.get("label_encoder")
	features = artifact.get("features")

	if model is None or le is None or features is None:
		print("Error: model artifact does not contain expected keys ('model','label_encoder','features').")
		sys.exit(1)

	df = pd.read_csv(csv_path)

	missing = [f for f in features if f not in df.columns]
	if missing:
		print(f"Error: input CSV is missing required feature columns: {missing}")
		sys.exit(1)

	X = df[features]
	preds_encoded = model.predict(X)
	try:
		preds = le.inverse_transform(preds_encoded)
	except Exception:
		# If inverse_transform fails, fallback to encoded values
		preds = preds_encoded

	df = df.copy()
	df["prediction"] = preds
	df.to_csv(out_path, index=False)
	print(f"Predictions written to: {out_path}")


def parse_args():
	p = argparse.ArgumentParser(description="Run predictions using a trained model artifact")
	p.add_argument("--csv", required=True, help="CSV with input rows to predict on")
	p.add_argument("--model", default="trained_model.pkl", help="Path to trained model artifact")
	p.add_argument("--out", default="predictions.csv", help="Output CSV path with predictions")
	return p.parse_args()


def main():
	args = parse_args()
	predict(args.csv, args.model, args.out)


if __name__ == "__main__":
	main()
