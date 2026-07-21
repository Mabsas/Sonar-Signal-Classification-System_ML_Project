"""
Sonar Signal Classification System
------------------------------------
Streamlit front-end. All ML/validation logic lives in utils/ -
this file is UI only.
"""

# ----------------------------------
# Load Libraries
# ----------------------------------
import io

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from utils.batch_prediction import predict_batch, validate_csv, normalize_uploaded_dataframe
from utils.prediction import feature_order, model, predict_sonar

# shap pulls in numba, which can conflict with newer numpy releases on
# some platforms. Import it lazily so a broken shap/numba install only
# disables explainability instead of crashing the whole app.
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ----------------------------------
# Load Artifacts
# ----------------------------------
feature_names = joblib.load("artifacts/feature_names.pkl")

# ----------------------------------
# Page Configuration
# ----------------------------------
st.set_page_config(
    page_title="Sonar Signal Classification System",
    page_icon="🌊",
    layout="wide",
)

# ----------------------------------
# Session State
# ----------------------------------
# Remembers the last single-sample input so the Explainability page
# can generate a local SHAP explanation for it without recomputation.
if "last_single_input" not in st.session_state:
    st.session_state["last_single_input"] = None

# ----------------------------------
# Sidebar
# ----------------------------------
with st.sidebar:

    st.title("📌 Project Info")

    st.markdown("---")
    page = st.radio(
        "📂 Navigation",
        [
            "About",
            "Single Prediction",
            "Batch Prediction",
            "Model Explainability",
        ],
    )
    st.markdown("---")

    st.write("### Project")
    st.write("**Sonar Signal Classification System**")
    st.caption("Rock vs Mine Prediction using Machine Learning")

    st.write("### Model")
    st.success("Tuned XGBoost Classifier")

    st.write("### Dataset")
    st.info("UCI Sonar Dataset")

    st.write("### Target Classes")
    st.write("🪨 Rock")
    st.write("💣 Mine")

    st.markdown("---")

    st.write("Developed by")
    st.write("**Md. A. B Siddique Alam**")
    st.caption("B.Sc. in CSE | AUST")




# =====================================================
# ABOUT PAGE
# =====================================================

if page == "About":

  st.title("🌊 Sonar Signal Classification")
  st.subheader("Rock vs Mine Prediction")

    st.markdown(
        """
The **Sonar Signal Classification System** is an end-to-end supervised machine learning project developed using the **UCI Sonar Dataset**. The objective is to classify sonar signals as either **Rock (R)** or **Mine (M)** based on 60 numerical sonar signal measurements collected from underwater objects.

This project was designed not only to build an accurate predictive model but also to demonstrate a complete machine learning development workflow. From exploratory data analysis and model comparison to hyperparameter optimization and explainable AI, every stage of the project follows practices commonly used in real-world machine learning applications.

---

## 📊 Dataset Overview

The project utilizes the **UCI Sonar Dataset**, which contains:

* **208 samples**
* **60 numerical sonar signal features**
* **Binary target variable**

  * **R → Rock**
  * **M → Mine**

Each feature represents the energy of a sonar signal reflected at different frequencies, making this a binary classification problem.

---

## 🔍 Exploratory Data Analysis (EDA)

Before training any models, a comprehensive exploratory analysis was performed to better understand the dataset.

The analysis included:

* Dataset inspection and statistical summaries
* Missing value and duplicate record analysis
* Class distribution analysis
* Feature distribution visualization
* Box plots for identifying feature spread and potential outliers
* Correlation heatmap to examine relationships between sonar features

These analyses provided valuable insights into the dataset and ensured the data was suitable for model development.

---

## ⚙️ Data Preprocessing

To prepare the data for machine learning, several preprocessing steps were carried out:

* Feature and target separation
* Label encoding of target classes
* Stratified train-test splitting
* Feature scaling for algorithms requiring standardized inputs
* Consistent feature ordering for reliable model inference

The preprocessing pipeline was designed to ensure identical transformations during both training and prediction.

---

## 🤖 Model Development

Multiple machine learning algorithms were implemented and evaluated to determine the most suitable classifier for the sonar dataset.

The models developed include:

* Logistic Regression
* Support Vector Machine (SVM)
* Decision Tree
* Random Forest
* XGBoost
* CatBoost

Rather than relying on a single algorithm, each model was trained and compared using identical datasets and evaluation criteria to ensure a fair comparison.

---

## 📈 Model Evaluation & Optimization

Model performance was assessed using a comprehensive set of evaluation metrics rather than accuracy alone.

The evaluation process included:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* Classification Report
* ROC Curve
* ROC-AUC Score
* Precision–Recall Curve
* Average Precision Score

To improve model robustness and reduce the risk of overfitting, **Stratified K-Fold Cross Validation** was performed. The models were further optimized using **GridSearchCV**, allowing multiple hyperparameter combinations to be evaluated systematically.

This experimentation demonstrated the impact of hyperparameter tuning and helped identify the most effective model configuration.

---

## 🧠 Explainable AI with SHAP

To improve model transparency and interpretability, **SHAP (SHapley Additive Explanations)** was incorporated into the project.

Several SHAP visualizations were generated to understand how different sonar signal features influenced model predictions, including:

* SHAP Summary Plot
* Feature Importance Plot
* Dependence Plot
* Waterfall Plot
* Force Plot

These explainability techniques provide insight into the decision-making process of the model, making its predictions more understandable and trustworthy.

---

## 🏆 Final Model Performance

After extensive experimentation and optimization, the **Tuned XGBoost Classifier** was selected as the final model due to its superior predictive performance and strong generalization capability.

### Final Results

| Metric                        |      Score |
| :---------------------------- | ---------: |
| **Test Accuracy**             | **92.86%** |
| **Cross-Validation Accuracy** | **87.53%** |
| **ROC-AUC Score**             |  **0.980** |
| **Average Precision Score**   |  **0.980** |

---

## 🎯 Skills Demonstrated

This project showcases practical experience with:

* Exploratory Data Analysis (EDA)
* Data Preprocessing
* Feature Engineering
* Binary Classification
* Machine Learning Model Comparison
* Cross-Validation Techniques
* Hyperparameter Optimization
* Ensemble Learning (Random Forest, XGBoost & CatBoost)
* Model Evaluation & Performance Analysis
* Explainable AI using SHAP
* Model Serialization
* Production-Ready Prediction Pipeline Development

---

## 📌 Project Summary

This project demonstrates the complete machine learning lifecycle—from understanding and exploring raw sonar data to developing, evaluating, optimizing, and interpreting predictive models. Through systematic experimentation, rigorous evaluation, and explainable AI techniques, the final **Tuned XGBoost** model achieved strong predictive performance while maintaining transparency and reliability. The result is a well-structured machine learning solution that reflects industry-standard practices in model development and evaluation.
"""
    )

# =====================================================
# SINGLE PREDICTION PAGE
# =====================================================

elif page == "Single Prediction":

    st.title("🌊 Sonar Signal Classification System")
    st.caption("Rock vs Mine Prediction")

    st.markdown(
        """
        This application predicts whether a sonar signal corresponds to a **Rock**
        or a **Mine** using a machine learning model trained on the UCI Sonar dataset.

        Enter the 60 sonar feature values and click **Predict**.
        """
    )

    st.markdown("---")
    st.header("📥 Enter Sonar Feature Values")
    st.caption(
        "Type each value exactly as you have it - any number of decimal "
        "places is accepted, nothing gets rounded."
    )

    col1, col2, col3 = st.columns(3)

    raw_inputs = []

    # First 20 features
    with col1:
        for i in range(20):
            raw_value = st.text_input(
                feature_names[i],
                value="0.0",
                key=f"feature_{i}",
            )
            raw_inputs.append(raw_value)

    # Next 20 features
    with col2:
        for i in range(20, 40):
            raw_value = st.text_input(
                feature_names[i],
                value="0.0",
                key=f"feature_{i}",
            )
            raw_inputs.append(raw_value)

    # Last 20 features
    with col3:
        for i in range(40, 60):
            raw_value = st.text_input(
                feature_names[i],
                value="0.0",
                key=f"feature_{i}",
            )
            raw_inputs.append(raw_value)

    st.markdown("---")

    if st.button("🔍 Predict", use_container_width=True):

        # -----------------------------
        # Parse raw text into floats, preserving exact typed precision
        # -----------------------------
        features = []
        parse_errors = []

        for name, raw_value in zip(feature_names, raw_inputs):
            try:
                features.append(float(raw_value.strip()))
            except (ValueError, AttributeError):
                parse_errors.append(name)

        if parse_errors:
            st.error(
                "❌ These fields contain a non-numeric value: "
                + ", ".join(parse_errors)
            )

        elif all(v == 0.0 for v in features):
            st.warning(
                "⚠️ Please enter sonar feature values before predicting."
            )

        else:
            out_of_range = [
                name for name, v in zip(feature_names, features) if not (0.0 <= v <= 1.0)
            ]
            if out_of_range:
                st.warning(
                    "⚠️ These values are outside the model's expected 0-1 "
                    "range and may produce unreliable predictions: "
                    + ", ".join(out_of_range)
                )

            try:
                prediction, confidence = predict_sonar(features)
                confidence = float(confidence)

                # Save for the Model Explainability page
                st.session_state["last_single_input"] = features

                st.markdown("---")
                st.subheader("📊 Prediction Result")

                result_col, confidence_col = st.columns(2)

                with result_col:
                    if prediction == "R":
                        st.success("🪨 Rock Detected")
                    else:
                        st.error("💣 Mine Detected")

                with confidence_col:
                    st.metric(
                        label="Confidence",
                        value=f"{confidence:.2f}%",
                    )

                st.progress(confidence / 100)
                st.caption("✅ Prediction completed successfully.")
                st.caption(
                    "ℹ️ Visit **Model Explainability** in the sidebar to see "
                    "which features drove this prediction."
                )

            except (ValueError, RuntimeError) as exc:
                st.error(f"❌ {exc}")

# =====================================================
# BATCH PREDICTION PAGE
# =====================================================

elif page == "Batch Prediction":

    st.title("📂 Batch CSV Prediction")

    st.write(
        """
        Upload a file containing **60 sonar signal features** per row.
        Each row represents one sonar sample.
        """
    )

    st.caption(
        "Need a template? A ready-to-use example is bundled with this app - "
        "view or download it below."
    )

    # -----------------------------
    # Sample Template Preview & Download
    # -----------------------------
    try:
        sample_df = pd.read_csv("sample_data/sample_batch.csv")

        with st.expander("👀 View sample_batch.csv"):
            st.dataframe(sample_df, use_container_width=True)

        with open("sample_data/sample_batch.csv", "rb") as sample_file:
            st.download_button(
                label="⬇️ Download sample_batch.csv",
                data=sample_file.read(),
                file_name="sample_batch.csv",
                mime="text/csv",
            )

    except FileNotFoundError:
        st.warning(
            "⚠️ `sample_data/sample_batch.csv` could not be found in this "
            "project. Please add it back to enable the sample template."
        )

    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=["csv", "xlsx", "xls"],
    )

    if uploaded_file is not None:

        try:
            if uploaded_file.name.lower().endswith((".xlsx", ".xls")):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)
        except Exception as exc:
            st.error(f"❌ Could not read the uploaded file: {exc}")
            st.stop()

        st.success("✅ File uploaded successfully!")

        # Auto-fix common formatting issues (e.g. missing header row)
        df, normalization_note = normalize_uploaded_dataframe(df)
        if normalization_note:
            st.info(normalization_note)

        # Display file information
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Rows", df.shape[0])

        with col2:
            st.metric("Columns", df.shape[1])

        st.subheader("📋 Preview")

        st.dataframe(
            df.head(),
            use_container_width=True,
        )

        st.markdown("---")
        st.subheader("🔍 Validation")

        is_valid, error_message = validate_csv(df)

        if not is_valid:
            st.error(f"❌ {error_message}")
            st.stop()

        st.success("✅ Validation Successful!")

        # -----------------------------
        # Run Predictions
        # -----------------------------
        st.markdown("---")
        st.subheader("🔮 Predictions")

        try:
            with st.spinner("Running predictions on all rows..."):
                results = predict_batch(df)
        except Exception as exc:
            st.error(f"❌ Prediction failed: {exc}")
            st.stop()

        rock_count = int((results["Prediction"] == "Rock").sum())
        mine_count = int((results["Prediction"] == "Mine").sum())

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Total Samples", len(results))
        metric_col2.metric("🪨 Rock", rock_count)
        metric_col3.metric("💣 Mine", mine_count)

        st.dataframe(
            results[["Prediction", "Confidence (%)"]],
            use_container_width=True,
        )

        with st.expander("View full results with original features"):
            st.dataframe(results, use_container_width=True)

        # -----------------------------
        # Download Results
        # -----------------------------
        csv_buffer = io.StringIO()
        results.to_csv(csv_buffer, index=False)

        st.download_button(
            label="⬇️ Download predictions.csv",
            data=csv_buffer.getvalue(),
            file_name="predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )

# =====================================================
# MODEL EXPLAINABILITY PAGE (SHAP)
# =====================================================

elif page == "Model Explainability":

    st.title("🧠 Model Explainability")
    st.write(
        """
        SHAP (SHapley Additive exPlanations) values show how much each
        sonar feature pushed the model's prediction toward **Rock** or
        **Mine**. This builds trust in the model by making its
        reasoning transparent instead of treating it as a black box.
        """
    )

    st.markdown("---")
    st.subheader("🌐 Global Feature Importance")
    st.caption(
        "Which features matter most to the model overall, "
        "based on the trained XGBoost classifier."
    )

    importance_df = (
        pd.DataFrame(
            {
                "Feature": feature_order,
                "Importance": model.feature_importances_,
            }
        )
        .sort_values("Importance", ascending=False)
        .head(15)
    )

    fig, ax = plt.subplots(figsize=(15, 8.5))
    ax.barh(importance_df["Feature"][::-1], importance_df["Importance"][::-1], color="#2E86AB")
    ax.set_xlabel("Importance")
    ax.set_title("Top 15 Most Important Sonar Features")
    fig.tight_layout()

    chart_col, _ = st.columns([2, 1])
    with chart_col:
        st.pyplot(fig, use_container_width=False)

    # -----------------------------
    # Local Explanation
    # -----------------------------
    st.markdown("---")
    st.subheader("🔎 Local Explanation (Last Single Prediction)")

    last_input = st.session_state.get("last_single_input")

    if not SHAP_AVAILABLE:
        st.warning(
            "⚠️ The `shap` package could not be loaded in this environment "
            "(often a numpy/numba version conflict), so per-prediction "
            "explanations are unavailable. The global feature importance "
            "chart above still works normally."
        )
    elif last_input is None or all(v == 0.0 for v in last_input):
        st.info(
            "ℹ️ Make a prediction on the **Single Prediction** page first "
            "to see a feature-by-feature explanation for that specific sample."
        )
    else:
        try:
            input_df = pd.DataFrame([last_input], columns=feature_order)

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df)

            shap_row = pd.Series(shap_values[0], index=feature_order)
            top_contributors = shap_row.reindex(
                shap_row.abs().sort_values(ascending=False).index
            ).head(10)

            colors = ["#E63946" if v > 0 else "#2E86AB" for v in top_contributors.values]

            fig2, ax2 = plt.subplots(figsize=(15, 8.5))
            ax2.barh(top_contributors.index[::-1], top_contributors.values[::-1], color=colors[::-1])
            ax2.set_xlabel("SHAP value (impact on model output)")
            ax2.set_title("Top 10 Features Driving This Prediction")
            ax2.axvline(0, color="black", linewidth=0.8)
            fig2.tight_layout()

            chart_col2, _ = st.columns([2, 1])
            with chart_col2:
                st.pyplot(fig2, use_container_width=False)

            st.caption(
                "🔴 Red bars push the prediction toward **Mine** · "
                "🔵 Blue bars push the prediction toward **Rock**."
            )

        except Exception as exc:
            st.warning(f"⚠️ Could not generate a SHAP explanation: {exc}")
