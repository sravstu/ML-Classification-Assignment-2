
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="ML Classification Dashboard",
    layout="wide"
)

st.title("Machine Learning Classification Dashboard")
st.write("Comparison of five classification algorithms using the Breast Cancer Wisconsin dataset.")

models = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "KNN": "knn",
    "Naive Bayes": "naive_bayes",
    "SVM": "svm"
}

st.header("Model Performance Comparison")

results = pd.read_csv("model_comparison.csv")
st.dataframe(results, use_container_width=True)

st.subheader("Accuracy Comparison")
st.bar_chart(results.set_index("ML Model Name")["Accuracy"])

st.header("Interactive Prediction")

selected = st.selectbox("Select Classification Model", list(models.keys()))

model_path = Path(models[selected] + ".joblib")
model = joblib.load(model_path)

test_data = pd.read_csv("test_data.csv")
features = test_data.drop(columns=["target"])

st.write("Select a test sample to predict:")

sample_index = st.slider(
    "Test Sample",
    min_value=0,
    max_value=len(features)-1,
    value=0
)

sample = features.iloc[[sample_index]]

st.dataframe(sample, use_container_width=True)

if st.button("Predict"):
    prediction = int(model.predict(sample)[0])
    probability = model.predict_proba(sample)[0]

    diagnosis = "Benign" if prediction == 1 else "Malignant"

    st.success(f"Predicted Class: {diagnosis}")
    st.write(f"Confidence: {max(probability)*100:.2f}%")

    actual = int(test_data.iloc[sample_index]["target"])
    actual_label = "Benign" if actual == 1 else "Malignant"
    st.write(f"Actual Class: {actual_label}")

st.caption("Educational demonstration only. Not for clinical diagnosis.")
