import streamlit as st
import pandas as pd
import numpy as np
import io
import pickle
import joblib
from typing import Optional


def load_model(file):
	"""Try joblib then pickle to load the uploaded model file.

	Returns (model, expected_features) where expected_features is a list of
	column names used during training (if available), otherwise None.
	"""
	def _extract(obj):
		# If user saved a dict with model + columns
		if isinstance(obj, dict):
			m = obj.get("model") or obj.get("estimator") or obj.get("est")
			cols = obj.get("columns") or obj.get("feature_names") or obj.get("features")
			if m is not None and cols is not None:
				return m, list(cols)
		# If estimator exposes feature names (sklearn >=1.0)
		if hasattr(obj, "feature_names_in_"):
			try:
				return obj, list(obj.feature_names_in_)
			except Exception:
				return obj, None
		return obj, None

	try:
		file.seek(0)
		obj = joblib.load(file)
		return _extract(obj)
	except Exception:
		try:
			file.seek(0)
			obj = pickle.load(file)
			return _extract(obj)
		except Exception:
			return None, None


def main():
	st.set_page_config(page_title="Model prediction app", layout="wide")
	st.title("Model Prediction App")

	st.markdown(
		"Upload a trained model file (`.pkl` or `.joblib`) and a CSV with feature rows to get predictions."
	)

	st.sidebar.header("Model & Data")
	model_file = st.sidebar.file_uploader("Upload model (.pkl or .joblib)", type=["pkl", "joblib"])
	data_file = st.sidebar.file_uploader("Upload input CSV (rows of features)", type=["csv"])
	st.sidebar.markdown("---")
	st.sidebar.info("CSV must contain the same feature columns the model expects.")

	col1 = st.columns([1])[0]

	model = None
	expected_features = None
	if model_file is not None:
		model, expected_features = load_model(model_file)
		if model is None:
			st.sidebar.error("Failed to load model. Ensure it's a pickle or joblib file created with compatible libraries.")
		else:
			st.sidebar.success("Model loaded successfully")
			if expected_features is not None:
				st.sidebar.write(f"Model expects {len(expected_features)} feature columns")

	with col1:
		st.subheader("Input data")
		df = None
		if data_file is not None:
			try:
				df = pd.read_csv(data_file)
				st.write(f"Loaded input CSV — shape: {df.shape}")
				st.dataframe(df)

				# Manual example input (like the notebook example) -------------------------------------------------
				st.markdown("---")
				st.markdown("### Manual example input (build raw features and predict)")
				with st.form(key='manual_example'):
					# Raw feature inputs — mirror the notebook's example fields
					country = st.text_input('country', value='USA')
					year = st.number_input('year', value=2023, step=1)
					industry = st.text_input('industry', value='IT')
					unemployment_rate = st.number_input('unemployment_rate', value=5.0, format="%f")
					job_openings = st.number_input('job_openings', value=1000.0, format="%f")
					remote_work_ratio = st.number_input('remote_work_ratio', value=50.0, format="%f")
					education_level = st.text_input('education_level', value='Master')
					tech_adoption_index = st.number_input('tech_adoption_index', value=0.8, format="%f")
					automation_risk = st.number_input('automation_risk', value=0.3, format="%f")
					gdp_growth_rate = st.number_input('gdp_growth_rate', value=2.5, format="%f")
					population_millions = st.number_input('population_millions', value=330.0, format="%f")
					demand_growth = st.number_input('demand_growth', value=5.0, format="%f")
					ai_influence_index = st.number_input('ai_influence_index', value=0.7, format="%f")
					trend_category = st.text_input('trend_category', value='Growing')
					submit_manual = st.form_submit_button('Predict example')
					if submit_manual:
						example = pd.DataFrame([{ 
							'country': country,
							'year': int(year),
							'industry': industry,
							'unemployment_rate': float(unemployment_rate),
							'job_openings': float(job_openings),
							'remote_work_ratio': float(remote_work_ratio),
							'education_level': education_level,
							'tech_adoption_index': float(tech_adoption_index),
							'automation_risk': float(automation_risk),
							'gdp_growth_rate': float(gdp_growth_rate),
							'population_millions': float(population_millions),
							'demand_growth': float(demand_growth),
							'ai_influence_index': float(ai_influence_index),
							'trend_category': trend_category
						}])
						# Preprocess like training: dummies then reindex to expected features if present
						proc_ex = pd.get_dummies(example, drop_first=True)
						if expected_features is not None:
							proc_ex = proc_ex.reindex(columns=expected_features, fill_value=0)
						# Predict
						try:
							if model is None:
								st.error('No model loaded. Upload a model first in the sidebar.')
							else:
								if hasattr(model, 'predict'):
									pred_val = model.predict(proc_ex)
									st.write('Predicted avg_salary_usd:')
									st.write(pred_val[0])
								else:
									st.error('Loaded model has no predict method')
						except Exception as e:
							st.error(f'Prediction failed: {e}')

				
			except Exception as e:
				st.error(f"Could not read CSV: {e}")
		else:
			st.info("Upload a CSV with feature columns for batch predictions.")


	# Removed the right-column 'Model info & tips' per user request.


if __name__ == "__main__":
	main()
