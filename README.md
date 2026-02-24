# 📱 Smartphone Usage Productivity Prediction App

This Streamlit application predicts user productivity level based on smartphone usage data.

## 🚀 Features
- Loads Smartphone Usage Dataset
- Preprocesses categorical features
- Trains Random Forest Model
- Takes user input
- Predicts Productivity Level
- Saves trained model

---

## 📂 Dataset Required

Place this file in the same folder:

Smartphone_Usage_Productivity_Dataset_50000.csv

---

## ▶️ Run Locally

1. Install dependencies:

pip install -r requirements.txt

2. Run the app:

streamlit run app.py

---

## 🌐 Deploy on Streamlit Cloud

1. Upload:
   - app.py
   - requirements.txt
   - dataset file

2. Click Deploy

---

## ⚠️ Important

If you get:

KeyError: 'Productivity'

Check column name using:

print(df.columns)

And update this line in app.py:

target_column = "Productivity"

---

## 👩‍💻 Author
Data Science ML Project
