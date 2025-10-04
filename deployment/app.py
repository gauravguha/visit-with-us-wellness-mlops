import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import requests, io

st.set_page_config(page_title="Visit With Us — Wellness Package Predictor", page_icon="🧳", layout="centered")

# Model artifact on Hugging Face Model Hub (your repo)
MODEL_URL = "https://huggingface.co/gauravguha/visit-with-us-wellness-model/resolve/main/model_pipeline.joblib?download=1"

@st.cache_resource(show_spinner=True)
def load_model():
    # Download model bytes and load directly from memory (no disk writes)
    r = requests.get(MODEL_URL, stream=True)
    r.raise_for_status()
    return load(io.BytesIO(r.content))

model = load_model()

st.title("Wellness Package Purchase — Prediction")
st.write("Fill the details and click **Predict**. The model estimates the probability that a customer will buy the Wellness Tourism Package.")

with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        Age = st.number_input("Age", min_value=18, max_value=90, value=35, step=1)
        CityTier = st.number_input("CityTier (1=metro, 2, 3)", min_value=1, max_value=3, value=1, step=1)
        DurationOfPitch = st.number_input("DurationOfPitch (minutes)", min_value=0.0, value=10.0, step=1.0)
        NumberOfPersonVisiting = st.number_input("NumberOfPersonVisiting", min_value=1.0, value=3.0, step=1.0)
        NumberOfFollowups = st.number_input("NumberOfFollowups", min_value=0.0, value=3.0, step=1.0)
        PreferredPropertyStar = st.number_input("PreferredPropertyStar (1-5)", min_value=1.0, max_value=5.0, value=3.0, step=1.0)

    with col2:
        NumberOfTrips = st.number_input("NumberOfTrips (per year)", min_value=0.0, value=2.0, step=1.0)
        Passport = st.selectbox("Passport", options=[0,1], index=1)
        PitchSatisfactionScore = st.number_input("PitchSatisfactionScore (1-5)", min_value=1.0, max_value=5.0, value=3.0, step=1.0)
        OwnCar = st.selectbox("OwnCar", options=[0,1], index=0)
        NumberOfChildrenVisiting = st.number_input("NumberOfChildrenVisiting (under 5)", min_value=0.0, value=0.0, step=1.0)
        MonthlyIncome = st.number_input("MonthlyIncome", min_value=0.0, value=25000.0, step=500.0)

    TypeofContact = st.selectbox("TypeofContact", ["Company Invited", "Self Enquiry"])
    Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Free Lancer"])
    Gender = st.selectbox("Gender", ["Male", "Female"])
    ProductPitched = st.selectbox("ProductPitched", ["Basic", "Deluxe", "Standard"])
    MaritalStatus = st.selectbox("MaritalStatus", ["Single", "Married", "Divorced", "Unmarried"])
    Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager"])

    submitted = st.form_submit_button("Predict")

if submitted:
    # NOTE: include ALL training features (excluding CustomerID and ProdTaken)
    row = {
        "Age": float(Age),
        "CityTier": float(CityTier),
        "DurationOfPitch": float(DurationOfPitch),
        "TypeofContact": str(TypeofContact).strip(),   # <-- added
        "Occupation": str(Occupation).strip(),
        "Gender": str(Gender).strip(),
        "NumberOfPersonVisiting": float(NumberOfPersonVisiting),
        "NumberOfFollowups": float(NumberOfFollowups),
        "ProductPitched": str(ProductPitched).strip(),
        "PreferredPropertyStar": float(PreferredPropertyStar),
        "MaritalStatus": str(MaritalStatus).strip(),
        "NumberOfTrips": float(NumberOfTrips),
        "Passport": float(Passport),
        "PitchSatisfactionScore": float(PitchSatisfactionScore),
        "OwnCar": float(OwnCar),
        "NumberOfChildrenVisiting": float(NumberOfChildrenVisiting),
        "Designation": str(Designation).strip(),
        "MonthlyIncome": float(MonthlyIncome),
    }
    X = pd.DataFrame([row])

    proba = model.predict_proba(X)[:, 1][0]
    pred  = int(proba >= 0.5)

    st.subheader("Result")
    st.metric("Predicted probability of purchase", f"{proba:.3f}")
    st.write("Prediction:", "**Yes**" if pred==1 else "**No**")

    with st.expander("Show input as model sees it"):
        st.dataframe(X)
