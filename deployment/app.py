import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# -------------------------------------------------
# Load Model and Preprocessing Artifacts
# -------------------------------------------------
repo_id = "ayush12358/tourism-xgb-model2"

# Model
model_path = hf_hub_download(
    repo_id=repo_id,
    filename="xgb_tourism_model.pkl",
    repo_type="model"
)
model = joblib.load(model_path)

# Feature names
feature_path = hf_hub_download(
    repo_id=repo_id,
    filename="feature_names.pkl",
    repo_type="model"
)
feature_names = joblib.load(feature_path)

# Label encoders
encoders_path = hf_hub_download(
    repo_id=repo_id,
    filename="label_encoders.pkl",
    repo_type="model"
)
label_encoders = joblib.load(encoders_path)

st.title("Tourism Product Adoption Prediction App")
st.write("""
This app predicts whether a customer is likely to purchase the tourism product
based on their demographic and travel-related information.
""")

# -------------------------------------------------
# User Input Form
# -------------------------------------------------
with st.form("prediction_form"):
    st.subheader("Enter Customer Details")

    # Numeric inputs
    age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
    duration_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=60, value=15)
    number_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
    number_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=2)
    preferred_star = st.selectbox("Preferred Property Star Rating", [1, 2, 3, 4, 5])
    number_trips = st.number_input("Number of Trips", min_value=0, max_value=50, value=2)
    monthly_income = st.number_input("Monthly Income", min_value=0, max_value=1000000, value=50000, step=1000)

    # Binary inputs
    passport = st.selectbox("Passport", [0, 1])  # 0 = No, 1 = Yes
    own_car = st.selectbox("Own Car", [0, 1])   # 0 = No, 1 = Yes

    # Categorical inputs (raw strings, will be encoded using saved LabelEncoders)
    typeof_contact = st.selectbox("Type of Contact", label_encoders['TypeofContact'].classes_.tolist())
    occupation = st.selectbox("Occupation", label_encoders['Occupation'].classes_.tolist())
    gender = st.selectbox("Gender", label_encoders['Gender'].classes_.tolist())
    product_pitched = st.selectbox("Product Pitched", label_encoders['ProductPitched'].classes_.tolist())
    marital_status = st.selectbox("Marital Status", label_encoders['MaritalStatus'].classes_.tolist())
    designation = st.selectbox("Designation", label_encoders['Designation'].classes_.tolist())

    submit = st.form_submit_button("Predict")

# -------------------------------------------------
# Prepare input data
# -------------------------------------------------
if submit:
    # Encode categorical inputs using the saved encoders
    input_dict = {
        "Age": age,
        "DurationOfPitch": duration_pitch,
        "NumberOfPersonVisiting": number_visiting,
        "NumberOfFollowups": number_followups,
        "PreferredPropertyStar": preferred_star,
        "NumberOfTrips": number_trips,
        "MonthlyIncome": monthly_income,
        "Passport": passport,
        "OwnCar": own_car,
        "TypeofContact": label_encoders['TypeofContact'].transform([typeof_contact])[0],
        "Occupation": label_encoders['Occupation'].transform([occupation])[0],
        "Gender": label_encoders['Gender'].transform([gender])[0],
        "ProductPitched": label_encoders['ProductPitched'].transform([product_pitched])[0],
        "MaritalStatus": label_encoders['MaritalStatus'].transform([marital_status])[0],
        "Designation": label_encoders['Designation'].transform([designation])[0],
    }

    input_data = pd.DataFrame([input_dict])

    # Align with training features
    input_data = input_data.reindex(columns=feature_names, fill_value=0)

    # Predict
    prediction = model.predict(input_data)[0]
    result = "Will Take the Product ✅" if prediction == 1 else "Will NOT Take the Product ❌"

    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
