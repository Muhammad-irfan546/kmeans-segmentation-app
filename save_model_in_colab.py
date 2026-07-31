# ============================================================
# RUN THIS IN YOUR COLAB NOTEBOOK, AFTER TRAINING YOUR MODEL
# This saves your KMeans model + scaler so the Streamlit app can load them
# ============================================================

import pickle

# --- Save the trained KMeans model ---
# Replace `kmeans` with the actual variable name of your trained model
with open("kmeans_model.pkl", "wb") as f:
    pickle.dump(kmeans, f)

# --- Save the fitted StandardScaler ---
# Replace `scaler` with the actual variable name of your fitted scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Saved kmeans_model.pkl and scaler.pkl")

# --- Download both files to your computer ---
from google.colab import files
files.download("kmeans_model.pkl")
files.download("scaler.pkl")
