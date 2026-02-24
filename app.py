import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Smartphone Productivity App")

st.title("📱 Smartphone Productivity Prediction")

# ==============================
# Load Dataset
# ==============================
DATA_PATH = "Smartphone_Usage_Productivity_Dataset_50000.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()  # remove extra spaces
    return df

df = load_data()

st.write("### Dataset Columns")
st.write(df.columns)

# ==============================
# AUTO Detect Target Column
# ==============================

possible_targets = [col for col in df.columns if "product" in col.lower()]

if len(possible_targets) == 0:
    st.error("❌ No productivity-related column found. Check dataset column names.")
    st.stop()

target_column = possible_targets[0]

st.success(f"✅ Target Column Detected: {target_column}")

# ==============================
# Encode Categorical
# ==============================

label_encoders = {}

for col in df.columns:
    if df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# ==============================
# Features & Target
# ==============================

X = df.drop(columns=[target_column])
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
# User Input
# ==============================

st.sidebar.header("Enter User Data")

input_data = {}

for col in X.columns:
    input_data[col] = st.sidebar.number_input(
        col,
        float(X[col].min()),
        float(X[col].max())
    )

input_df = pd.DataFrame([input_data])

# ==============================
# Prediction
# ==============================

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    st.success(f"🎯 Predicted Productivity: {prediction}")
