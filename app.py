import streamlit as st
import pandas as pd
import joblib
import warnings
import matplotlib.pyplot as plt
import io

from src.data_cleaning import clean_data
from src.feature_engineering import encode_data
from src.model_training import train_models
from src.model_evaluation import evaluate_models

# Settings
st.set_page_config(page_title="AutoML App", layout="wide")
warnings.filterwarnings("ignore")
st.caption("Built by Ashen Caldera • AutoML Project")
st.divider()
st.sidebar.title("⚙️ Settings")
st.sidebar.write("Upload dataset and run AutoML")
if st.sidebar.button("🔄 Reset App"):
    st.session_state.clear()
    st.rerun()
st.markdown("""
# 🚀 AutoML Dashboard
### Train models • Compare • Predict • Download
""")

# =========================
# 📂 Upload Dataset
# =========================
st.markdown("## 📂 Step 1: Upload Training Data")

file = st.file_uploader("Upload your dataset", type=["csv"])

if file is not None:
    try:
        df = pd.read_csv(file, na_values='?')
        df.columns = df.columns.str.strip()

        if df.empty:
            st.error("❌ Uploaded file is empty!")
        else:
            st.success("✅ File loaded successfully")

            st.write("Dataset Preview:")
            st.dataframe(df.head())

            st.subheader("📊 Data Summary")
            st.write(df.describe())

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")

    target = st.selectbox("🎯 Select Target Column", df.columns)

    # =========================
    # 🚀 Run AutoML
    # =========================
    st.markdown("## ⚙️ Step 2: Train Models")


    if st.button("🚀 Run AutoML"):

        with st.spinner("Training models... ⏳"):

            df_clean = clean_data(df)
            df_encoded = encode_data(df_clean)

            st.session_state["processed_data"] = df_encoded
            
        if "processed_data" in st.session_state:

            st.subheader("📥 Download Processed Dataset")

            csv_processed = st.session_state["processed_data"].to_csv(index=False).encode("utf-8")

            st.download_button(
                label="📥 Download Cleaned & Encoded Dataset",
                data=csv_processed,
                file_name="processed_dataset.csv",
                mime="text/csv"
            )

            X = df_encoded.drop(columns=[target])
            y = df_encoded[target]

        # Detect problem type
            if y.nunique() < 20:
                problem_type = "classification"
            else:
                problem_type = "regression"

            st.write(f"Detected Problem Type: {problem_type}")
            models, X_test, y_test = train_models(X, y)

            results, best_model, best_name, best_score, problem_type = evaluate_models(
                models, X_test, y_test
            )

            # Save to session
            st.session_state["results"] = results
            st.session_state["best_model"] = best_model
            st.session_state["best_name"] = best_name
            st.session_state["best_score"] = best_score
            st.session_state["X_columns"] = X.columns.tolist()
            st.session_state["problem_type"] = problem_type


    # =========================
    # 📊 Show Results
    # =========================
    st.markdown("## 📊 Step 3: Results")
    if "results" in st.session_state:

        results = st.session_state["results"]
        best_model = st.session_state["best_model"]
        best_name = st.session_state["best_name"]
        best_score = st.session_state["best_score"]
        problem_type = st.session_state["problem_type"]

        results_df = pd.DataFrame(results).T

        if problem_type == "classification":
            results_df = results_df.rename(columns={
                "accuracy": "Accuracy",
                "f1": "F1 Score"
            })
        else:
            results_df = results_df.rename(columns={
                "rmse": "RMSE",
                "r2": "R2 Score"
            })
        st.success("✅ Training completed successfully!")
        st.dataframe(results_df.style.highlight_max(axis=0))

        st.subheader("🏆 Best Model")
        st.metric("Model Name", best_name)
        if problem_type == "classification":
            st.metric("F1 Score", f"{best_score:.4f}")
        else:
            st.metric("R2 Score", f"{best_score:.4f}")

        st.subheader("🧠 Insights")

        best_f1 = best_score

        if best_f1 > 0.85:
            insight = "Excellent model performance. Suitable for real-world deployment."
        elif best_f1 > 0.75:
            insight = "Good performance. Model can be improved with tuning."
        elif best_f1 > 0.60:
            insight = "Moderate performance. Consider feature engineering."
        else:
            insight = "Low performance. Data quality or model choice needs improvement."

        st.info(f"""
        **Best Model:** {best_name}  

        **Why?** It achieved the highest F1 Score of {best_f1:.4f}, balancing precision and recall effectively.

        **Assessment:** {insight}
        """)


        st.subheader("📊 Model Performance Chart")

        fig, ax = plt.subplots(figsize=(2, 2))  # 👈 smaller

        model_names = list(results.keys())

        if problem_type == "classification":
            scores = [results[m]["f1"] for m in model_names]
            ylabel = "F1 Score"
        else:
            scores = [results[m]["r2"] for m in model_names]
            ylabel = "R2 Score"

        ax.bar(model_names, scores)
        ax.set_ylabel(ylabel)
        ax.set_title("Model Comparison")

        plt.xticks(rotation=30)

        # 👇 IMPORTANT: disable container width
        st.pyplot(fig, use_container_width=False)

        # =========================
        # 📥 Download Model
        # =========================
        import io
        import joblib

        buffer = io.BytesIO()
        joblib.dump(best_model, buffer)
        buffer.seek(0)

        st.download_button(
            label="📥 Download Best Model",
            data=buffer,
            file_name="best_model.pkl",
            mime="application/octet-stream"
        )

if "results" in st.session_state:

    results = st.session_state["results"]
    best_model = st.session_state["best_model"]

    # existing results UI here...

    st.subheader("📌 Feature Importance")

    model = best_model

    if hasattr(model, "feature_importances_"):
        import matplotlib.pyplot as plt

        importances = model.feature_importances_
        features = st.session_state["X_columns"]

        fig, ax = plt.subplots(figsize=(4, 2.5))
        ax.barh(features, importances)

        st.pyplot(fig, use_container_width=False)
    else:
        st.info("Feature importance not available for this model")

# =========================
# 📂 Batch Prediction
# =========================

st.divider()
st.subheader("📂 Batch Prediction")

predict_file = st.file_uploader(
    "Upload new data for prediction", 
    type=["csv"], 
    key="predict"
)

if predict_file is not None:

    if "best_model" not in st.session_state:
        st.warning("⚠️ Please run AutoML first!")
    else:
        try:
            new_df = pd.read_csv(predict_file)
            new_df.columns = new_df.columns.str.strip()

            st.write("Preview of uploaded data:")
            st.dataframe(new_df.head())

            expected_cols = st.session_state["X_columns"]

            # Check missing columns
            missing_cols = set(expected_cols) - set(new_df.columns)

            if missing_cols:
                st.error(f"❌ Missing columns: {missing_cols}")
            else:
                # =========================
                # 🔹 Prepare Input Data (SAFE)
                # =========================
                input_df = new_df.copy()
                input_df = input_df[expected_cols]
                input_df = encode_data(input_df)

                # =========================
                # 🔹 Prediction
                # =========================
                preds = st.session_state["best_model"].predict(input_df)

                result_df = new_df.copy()
                result_df["Prediction"] = preds

                st.subheader("✅ Predictions")
                st.dataframe(result_df)

                # =========================
                # 🔍 Confidence Score
                # =========================
                problem_type = st.session_state["problem_type"]

                if (
                    problem_type == "classification"
                    and hasattr(st.session_state["best_model"], "predict_proba")
                ):
                    probs = st.session_state["best_model"].predict_proba(input_df)
                    confidence = probs.max(axis=1)

                    result_df["Confidence"] = confidence

                    st.subheader("🔍 Prediction Confidence")
                    st.dataframe(result_df[["Prediction", "Confidence"]])

                # =========================
                # 📊 Prediction Distribution Chart
                # =========================
                import matplotlib.pyplot as plt

                st.subheader("📊 Prediction Distribution")

                fig, ax = plt.subplots(figsize=(4, 2.5))

                if problem_type == "classification":
                    result_df["Prediction"].value_counts().plot(kind="bar", ax=ax)
                    ax.set_ylabel("Count")
                else:
                    result_df["Prediction"].plot(kind="hist", bins=20, ax=ax)
                    ax.set_ylabel("Frequency")

                ax.set_title("Prediction Distribution")

                st.pyplot(fig, use_container_width=False)

                # =========================
                # 🧠 Auto Insights
                # =========================
                st.subheader("🧠 Insights")

                if problem_type == "classification":
                    counts = result_df["Prediction"].value_counts()
                    dominant_class = counts.idxmax()

                    st.info(
                        f"Most predictions belong to class: {dominant_class} "
                        f"({counts.max()} records)"
                    )
                else:
                    avg_pred = result_df["Prediction"].mean()

                    st.info(
                        f"Average predicted value: {avg_pred:.2f}"
                    )

                # =========================
                # 📥 Download Predictions
                # =========================
                csv = result_df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="📥 Download Predictions",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Error: {e}")