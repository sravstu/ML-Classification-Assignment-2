# ML Classification Dashboard

## 1. Problem Statement
Compare five classification models on the same dataset
and demonstrate their predictions using Streamlit.

## 2. Dataset Description
Breast Cancer Wisconsin (Diagnostic)
Source: UCI Machine Learning Repository
https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

Instances: 569
Features: 30
Classes: Malignant and Benign
Train/Test split: 80/20, stratified

## 3. GitHub Repository Link
https://github.com/sravstu/ML-Classification-Assignment-2

## 4. Models Used and Evaluation Metrics
Logistic Regression, Decision Tree, KNN,
Gaussian Naive Bayes, and SVM.

| ML Model Name       |   Accuracy |    AUC |   Precision |   Recall |     F1 |    MCC |
|:--------------------|-----------:|-------:|------------:|---------:|-------:|-------:|
| Logistic Regression |     0.9825 | 0.9954 |      0.9861 |   0.9861 | 0.9861 | 0.9623 |
| Decision Tree       |     0.9123 | 0.9157 |      0.9559 |   0.9028 | 0.9286 | 0.8174 |
| KNN                 |     0.9561 | 0.9788 |      0.9589 |   0.9722 | 0.9655 | 0.9054 |
| Naive Bayes         |     0.9386 | 0.9878 |      0.9452 |   0.9583 | 0.9517 | 0.8676 |
| SVM                 |     0.9825 | 0.995  |      0.9861 |   0.9861 | 0.9861 | 0.9623 |

Model Performance Observations

- Logistic Regression: Achieved 98.25% accuracy and the highest AUC of 0.9954.
- Decision Tree: Had the lowest accuracy at 91.23% and the lowest MCC at 0.8174.
- KNN: Performed well with 95.61% accuracy and 97.22% recall.
- Naive Bayes: Achieved 93.86% accuracy and a relatively high AUC of 0.9878.
- SVM: Matched Logistic Regression's 98.25% accuracy, with an AUC of 0.9950.
Overall observation: Logistic Regression and SVM tied for the highest accuracy. Logistic Regression had a marginally higher AUC, making it a reasonable preferred model for this comparison.

## 5. Streamlit Application
https://ml-classification-assignment-2-cp2ccmnwjli45rdjabxykv.streamlit.app

The app displays evaluation results and allows users
to select a model and predict a test sample.

## 6. Execution
pip install -r requirements.txt
streamlit run app.py

## 7. Conclusion
All five models were evaluated on the same test set.
Logistic Regression and SVM achieved the highest accuracy.
