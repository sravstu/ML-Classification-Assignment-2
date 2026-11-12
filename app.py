
import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
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


st.divider()
st.header("Upload Test Data and Evaluate Model")

uploaded_file = st.file_uploader(
    "Upload test_data.csv",
    type=["csv"]
)

if uploaded_file is not None:
    uploaded_data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Test Data")
    st.dataframe(uploaded_data.head())

    if "target" not in uploaded_data.columns:
        st.error("CSV must contain a target column.")
    else:
        X_uploaded = uploaded_data.drop(columns=["target"])
        y_uploaded = uploaded_data["target"]

        model_path = Path("model") / f"{models[selected]}.joblib"

        if not model_path.exists():
            model_path = Path(f"{models[selected]}.joblib")

        if not model_path.exists():
            st.error("Saved model file not found.")
        else:
            loaded_model = joblib.load(model_path)

            if st.button("Evaluate Uploaded Data"):
                predictions = loaded_model.predict(X_uploaded)

                st.subheader("Classification Report")
                report = classification_report(
                    y_uploaded,
                    predictions,
                    zero_division=0
                )
                st.text(report)

                st.subheader("Confusion Matrix")
                matrix = confusion_matrix(
                    y_uploaded,
                    predictions,
                    labels=[0, 1]
                )

                st.dataframe(
                    pd.DataFrame(
                        matrix,
                        index=["Actual 0", "Actual 1"],
                        columns=["Predicted 0", "Predicted 1"]
                    )
                )

