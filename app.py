import streamlit as st
import numpy as np
import pickle

# Load saved model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("❤️ Heart Disease Prediction App")

st.write("Enter patient details below:")

# Input fields (based on heart dataset)
age = st.number_input("Age")
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure")
chol = st.number_input("Cholesterol")
fbs = st.selectbox("Fasting Blood Sugar > 120 (1 = Yes, 0 = No)", [0, 1])
restecg = st.selectbox("Rest ECG (0-2)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved")
exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])
oldpeak = st.number_input("ST Depression")
slope = st.selectbox("Slope (0-2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0-4)", [0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia (0-3)", [0, 1, 2, 3])

# Collect features
features = np.array([[age, sex, cp, trestbps, chol, fbs,
                      restecg, thalach, exang, oldpeak,
                      slope, ca, thal]])

# Prediction
if st.button("Predict"):
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)

    if prediction[0] == 1:
        st.error("⚠️ Heart Disease Detected")
    else:
        st.success("✅ No Heart Disease")