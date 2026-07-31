import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ============================================================
# EDIT THIS LIST: put your actual feature/column names here,
# in the same order you used when training the model
# ============================================================
FEATURE_NAMES = ["Quantity", "UnitPrice", "TotalPrice"]

# ------------------------------------------------------------
# Load model + scaler (cached so it only loads once)
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    with open("kmeans_model.pkl", "rb") as f:
        kmeans = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return kmeans, scaler

kmeans, scaler = load_model()

# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------
st.set_page_config(page_title="K-Means Segmentation", page_icon="🔍")
st.title("🔍 K-Means Clustering App")
st.write("Enter values below to see which cluster they belong to.")

# ------------------------------------------------------------
# Sidebar: choose single input or CSV upload
# ------------------------------------------------------------
mode = st.sidebar.radio("Input method", ["Manual input", "Upload CSV"])

if mode == "Manual input":
    st.subheader("Enter feature values")

    user_input = {}
    for feature in FEATURE_NAMES:
        user_input[feature] = st.number_input(feature, value=0.0)

    if st.button("Predict Cluster"):
        input_df = pd.DataFrame([user_input])
        scaled_input = scaler.transform(input_df)
        cluster = kmeans.predict(scaled_input)[0]
        st.success(f"Predicted Cluster: **{cluster}**")

else:
    st.subheader("Upload a CSV file")
    st.caption(f"CSV must contain these columns: {', '.join(FEATURE_NAMES)}")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        missing_cols = [c for c in FEATURE_NAMES if c not in data.columns]
        if missing_cols:
            st.error(f"Missing columns in CSV: {missing_cols}")
        else:
            scaled_data = scaler.transform(data[FEATURE_NAMES])
            data["Cluster"] = kmeans.predict(scaled_data)

            st.write("Results:")
            st.dataframe(data)

            csv_out = data.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download results as CSV",
                csv_out,
                "clustered_results.csv",
                "text/csv",
            )

# ------------------------------------------------------------
# Optional: show cluster centers
# ------------------------------------------------------------
with st.expander("View cluster centers"):
    centers = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=FEATURE_NAMES,
    )
    st.dataframe(centers)
