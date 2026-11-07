import streamlit as st
import pandas as pd
import numpy as np
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Bank Marketing Classification")
st.write(
    "Predict whether a customer will subscribe "
    "to a bank term deposit."
)

st.sidebar.header("Model Configuration")

model_options = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Support Vector Machine": "support_vector_machine.pkl"
}

selected_model = st.sidebar.selectbox(
    "Select Classification Model",
    list(model_options.keys())
)

st.subheader("Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload test CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.success("Dataset uploaded successfully!")

    st.write("Dataset Preview")
    st.dataframe(data.head())

    if "y" not in data.columns:
        st.error("Target column 'y' is missing.")

    else:

        X_test = data.drop(columns=["y"])

        y_test = data["y"].map({
            "no": 0,
            "yes": 1
        })

        if y_test.isnull().any():
            st.error("Target column must contain yes/no values.")

        else:

            model_path = (
                "model/" + model_options[selected_model]
            )

            try:
                model = joblib.load(model_path)

                y_pred = model.predict(X_test)

                if hasattr(model, "predict_proba"):
                    y_score = model.predict_proba(
                        X_test
                    )[:, 1]
                else:
                    y_score = model.decision_function(
                        X_test
                    )

                accuracy = accuracy_score(
                    y_test, y_pred
                )

                auc = roc_auc_score(
                    y_test, y_score
                )

                precision = precision_score(
                    y_test, y_pred,
                    zero_division=0
                )

                recall = recall_score(
                    y_test, y_pred,
                    zero_division=0
                )

                f1 = f1_score(
                    y_test, y_pred,
                    zero_division=0
                )

                mcc = matthews_corrcoef(
                    y_test, y_pred
                )

                st.header("Model Evaluation")

                st.write(
                    "Selected Model:", selected_model
                )

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Accuracy", f"{accuracy:.4f}"
                )

                col2.metric(
                    "AUC Score", f"{auc:.4f}"
                )

                col3.metric(
                    "Precision", f"{precision:.4f}"
                )

                col4, col5, col6 = st.columns(3)

                col4.metric(
                    "Recall", f"{recall:.4f}"
                )

                col5.metric(
                    "F1 Score", f"{f1:.4f}"
                )

                col6.metric(
                    "MCC", f"{mcc:.4f}"
                )

                st.subheader("Confusion Matrix")

                cm = confusion_matrix(
                    y_test, y_pred
                )

                fig, ax = plt.subplots()

                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=["No", "Yes"],
                    yticklabels=["No", "Yes"],
                    ax=ax
                )

                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")

                st.pyplot(fig)

                st.subheader("Classification Report")

                report = classification_report(
                    y_test,
                    y_pred,
                    target_names=["No", "Yes"],
                    output_dict=True,
                    zero_division=0
                )

                st.dataframe(
                    pd.DataFrame(report).transpose()
                )

            except Exception as e:
                st.error(f"Error loading or evaluating model: {e}")

else:
    st.info(
        "Please upload test_data.csv to view model results."
    )