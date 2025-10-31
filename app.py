# -*- coding: utf-8 -*-


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

st.set_page_config(page_title="Job Market Trend Predictor", layout="wide")

st.title("📊 Job Market Trend Analysis & Prediction")

# --- Upload CSV ---
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    st.subheader("Data Preview")
    st.dataframe(df.head())

    # --- Cleaning ---
    st.markdown("### 🧹 Cleaning the Data")
    df.drop_duplicates(inplace=True)
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns

    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    st.write("✅ Missing values handled.")
    st.write(f"Remaining missing values: {df.isnull().sum().sum()}")

    # --- Target Encoding ---
    st.markdown("### 🎯 Encode Target Column")
    target_col = st.selectbox("Select target column", df.columns)
    le = LabelEncoder()
    try:
        # encode into a new column; catch errors to show to user instead of hanging
        df["target_encoded"] = le.fit_transform(df[target_col].astype(str))
        st.write(f"Target column `{target_col}` encoded. Classes: {list(le.classes_)[:10]}{('...' if len(le.classes_)>10 else '')}")
    except Exception as e:
        st.exception(e)
        st.stop()

    # --- Visualization ---
    st.markdown("### 📈 Data Visualization")
    col = st.selectbox("Select column to visualize", df.columns)
    # Use the actual series for color to avoid ambiguity between column name and value
    try:
        # For very large datasets or many classes, sampling and disabling automatic
        # color-by-target avoids long renders and browser hangs.
        sample_n = min(len(df), 10000)
        df_plot = df.sample(n=sample_n, random_state=42) if len(df) > sample_n else df

        n_classes = df[target_col].nunique()
        color_by_target = True
        if n_classes > 50:
            color_by_target = False
            st.warning(f"Target has {n_classes} unique values — disabling automatic color-by-target to avoid slow plotting.")
            if st.checkbox(f"Enable color by target anyway (show {n_classes} classes)?", value=False):
                color_by_target = True

        with st.spinner("Creating visualization..."):
            if color_by_target:
                fig = px.histogram(df_plot, x=col, color=df_plot[target_col])
            else:
                fig = px.histogram(df_plot, x=col)

        # Provide a unique key to avoid StreamlitDuplicateElementId when rerunning
        # Replace deprecated `use_container_width` with `width='stretch'`.
        st.plotly_chart(fig, width='stretch', key=f"plot_{col}_{target_col}")
    except Exception as e:
        st.error(f"Could not create visualization: {e}")
        st.write(df[[col, target_col]].head())
    

    # --- Modeling ---
    st.markdown("### 🤖 Train a Model")
    # Build options and safe defaults for the multiselect so defaults are always valid
    feature_options = [c for c in df.columns if c not in [target_col, "target_encoded"]]
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c in feature_options]
    default_features = numeric_cols[:3]
    features = st.multiselect(
        "Select features for training",
        feature_options,
        default=default_features,
        key=f"features_{target_col}",
    )

    # Controls to limit resource usage on hosted runners
    max_train_samples = st.number_input(
        "Max training rows (sample if dataset is larger)",
        min_value=100,
        max_value=len(df) if len(df) > 0 else 100,
        value=min(5000, len(df)),
        step=100,
        help="If your dataset is larger than this, we'll randomly sample down before training to reduce memory/CPU usage on hosted platforms.",
    )

    reuse_model = st.checkbox("Reuse trained model in this session if available", value=True)

    if st.button("Train Model"):
        if not features:
            st.error("Please select at least one feature for training.")
        else:
            try:
                X = df[features]
                y = df["target_encoded"]

                # If dataset is large, sample to the requested max size before train/test split
                if len(X) > int(max_train_samples):
                    st.warning(f"Dataset has {len(X)} rows — sampling down to {int(max_train_samples)} rows for training.")
                    sample_idx = np.random.RandomState(42).choice(X.index, size=int(max_train_samples), replace=False)
                    X = X.loc[sample_idx].reset_index(drop=True)
                    y = y.loc[sample_idx].reset_index(drop=True)

                # If user has a trained model in this session and chose to reuse it, skip retraining
                if reuse_model and st.session_state.get("trained_artifact") is not None:
                    artifact = st.session_state["trained_artifact"]
                    model = artifact["model"]
                    le = artifact["label_encoder"]
                    features = artifact["features"]
                    st.info("Using previously trained model from this session.")
                    # Still evaluate on a holdout sample
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                else:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Reduce resource usage on hosted environments
                model = RandomForestClassifier(random_state=42, n_estimators=50, n_jobs=1)
                try:
                    with st.spinner("Training model..."):
                        model.fit(X_train, y_train)
                except MemoryError:
                    st.error("Training failed due to insufficient memory on the hosted environment. Try lowering 'Max training rows' or reducing model size (fewer trees).")
                    raise

                preds = model.predict(X_test)
                st.success("✅ Model trained successfully!")

                st.text("Classification Report:")
                # Only include labels that appear in the test set to avoid
                # a mismatch between the number of classes and target_names
                labels = np.unique(y_test)
                try:
                    target_names = le.inverse_transform(labels)
                except Exception:
                    # Fallback: use stringified labels if inverse transform fails
                    target_names = [str(l) for l in labels]

                st.text(classification_report(y_test, preds, labels=labels, target_names=target_names))

                st.text("Confusion Matrix:")
                st.write(confusion_matrix(y_test, preds, labels=labels))

                # Save artifact compatible with predict.py
                artifact = {"model": model, "label_encoder": le, "features": features}
                try:
                    # Use pickle.dumps to create bytes directly (avoids repeated disk writes
                    # or filesystem race conditions on hosted platforms).
                    model_bytes = pickle.dumps(artifact)
                    # Store the trained artifact in session_state for reuse during this session
                    try:
                        st.session_state["trained_artifact"] = artifact
                    except Exception:
                        # Session state may not be available in some contexts; ignore silently
                        pass
                    st.download_button("📥 Download Trained Model", data=model_bytes, file_name="trained_model.pkl")
                    st.success("Saved artifact to in-memory bytes for download.")
                except Exception as e:
                    st.exception(e)
            except Exception as e:
                st.exception(e)

else:
    st.info("👆 Upload a CSV file to get started.")
