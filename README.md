# Simple Streamlit App

This repository contains a minimal Streamlit app in `app.py` that supports:

- CSV upload and preview with descriptive statistics
- Image upload and preview
- A small text transform/count tool

## Quick start (local)

1. Create and activate a virtual environment (optional but recommended):

   ```powershell
   python -m venv .venv; .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

3. Run the app:

   ```powershell
   streamlit run app.py
   ```

## Deploy

- You can deploy this app to Streamlit Community Cloud. Push the repo to GitHub and connect it in the Streamlit Cloud dashboard.
- Make sure `requirements.txt` is present so Streamlit installs the dependencies.

## Notes

- The app is intentionally small and easy to extend. Add more processing or visualizations as needed.
