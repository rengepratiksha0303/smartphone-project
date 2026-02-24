# ==============================
# app.py
# ==============================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ==============================
# Page Config
# ==============================
st.set_page_config(page_title="Smartphone Productivity Predictor")

st.title("📱 Smartphone Usage Productivity Prediction")

# ==============================
# Load Dataset
# ==============================
DATA_PATH = "Smartphone_Usage_Productivity_Dataset_50000.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

df = load_data()

st.write("Dataset Preview")
st.dataframe(df.head())

# ==============================
# Preprocessing
# ==============================

label_encoders = {}

for col in df.columns:
    if df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# ⚠️ CHANGE this if your target column name is different
target_column = "Productivity"

X = df.drop(target_column, axis=1)
y = df[target_column]

# ==============================
# Train Model
# ==============================

@st.cache_resource
def train_model():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = train_model()

# ==============================
# User Input Section
# ==============================

st.sidebar.header("Enter User Data")

input_data = {}

for col in X.columns:
    if df[col].nunique() <= 10:
        input_data[col] = st.sidebar.selectbox(col, df[col].unique())
    else:
        input_data[col] = st.sidebar.number_input(col, float(df[col].min()), float(df[col].max()))

input_df = pd.DataFrame([input_data])

# ==============================
# Prediction
# ==============================

if st.button("Predict Productivity"):

    prediction = model.predict(input_df)[0]

    st.success(f"Predicted Productivity Level: {prediction}")

# ==============================
# Save Model (Optional)
# ==============================

if st.button("Save Model"):
    with open("productivity_model.pkl", "wb") as f:
        pickle.dump(model, f)
    st.success("Model saved as productivity_model.pkl")
