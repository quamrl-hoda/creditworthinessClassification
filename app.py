import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import plotly.graph_objects as go
import plotly.express as px

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(
    page_title="Premium Credit Score Dashboard",
    page_icon="💳",
    layout="wide",
)

# ------------------------------
# LOAD MODELS
# ------------------------------
@st.cache_resource
def load_models():
    logistic_model = joblib.load("model/logistic_pipeline.joblib")
    rf_model = joblib.load("model/rf_pipeline.joblib")

    feature_cols = logistic_model.named_steps['preprocessor'].feature_names_in_

    return logistic_model, rf_model, feature_cols

log_model, rf_model, feature_cols = load_models()

# ------------------------------
# PREMIUM HEADER
# ------------------------------
st.markdown("""
    <style>
        .glass-card {
            background: rgba(255, 255, 255, 0.12);
            padding: 25px;
            border-radius: 18px;
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.25);
            box-shadow: 0 8px 32px rgba(0,0,0,0.15);
        }
    </style>

    <h1 style="text-align:center; color:#4CAF50; font-size:40px;">💳 Premium Credit Score Prediction Dashboard</h1>
    <p style="text-align:center; font-size:18px; color:gray;">
        A beautifully designed ML-powered risk assessment tool.
    </p>
""", unsafe_allow_html=True)

# ------------------------------
# SIDEBAR NAVIGATION
# ------------------------------
menu = st.sidebar.radio(
    "🔍 Navigate",
    ["Predict Manually", "Upload CSV", "Data Explorer", "Feature Importance"],
)

model_choice = st.sidebar.selectbox(
    "🎯 Select ML Model",
    ("Logistic Regression", "Random Forest")
)

model = log_model if model_choice == "Logistic Regression" else rf_model

st.sidebar.markdown("---")
st.sidebar.caption("✨ Designed by Quamrul — AI & ML Dashboard")

# =========================================================================================
# 1️⃣ MANUAL PREDICTION
# =========================================================================================
if menu == "Predict Manually":
    st.subheader("📝 Enter Customer Information")

    numeric_keywords = ["age", "income", "amount", "num", "balance", "duration"]

    col1, col2, col3 = st.columns(3)
    input_data = {}

    for idx, col in enumerate(feature_cols):
        target = [col1, col2, col3][idx % 3]

        if any(k in col.lower() for k in numeric_keywords):
            input_data[col] = target.number_input(col, value=0.0)
        else:
            input_data[col] = target.text_input(col, value="")

    df = pd.DataFrame([input_data])[feature_cols]

    if st.button("🔮 Predict Credit Score", use_container_width=True):
        with st.spinner("Running advanced ML model..."):
            time.sleep(1.5)

        pred = model.predict(df)[0]

        # ------------------ Gauge Meter ------------------
        gauge_val = 20 if pred == "Poor" else 60 if pred == "Standard" else 90

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=gauge_val,
                gauge={'axis': {'range':[0,100]},
                       'bar': {'color': "#4CAF50"},
                       'steps': [
                           {'range': [0, 40], 'color':'#ff4d4d'},
                           {'range': [40, 70], 'color':'#ffd633'},
                           {'range': [70, 100], 'color':'#4CAF50'},
                       ]},
                title={'text': "Creditworthiness Score"}
            )
        )

        # ------------------ Output UI ------------------
        st.markdown("### 🎯 Prediction Result")
        st.success(f"**Predicted Category: `{pred}`**")

        st.plotly_chart(fig, use_container_width=True)

        # Probability chart
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(df)[0]
            st.markdown("### 📊 Probability Distribution")
            prob_df = pd.DataFrame(
                {"Class": [f"Class {i}" for i in range(len(probs))], "Probability": probs}
            )
            st.bar_chart(prob_df, x="Class", y="Probability")

        with st.expander("🧾 View Input Summary"):
            st.json(input_data)


# =========================================================================================
# 2️⃣ CSV UPLOAD PREDICTION
# =========================================================================================
elif menu == "Upload CSV":
    st.subheader("📂 Upload CSV File for Batch Prediction")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df_csv = pd.read_csv(file)
        st.write("### 🔍 Dataset Preview")
        st.dataframe(df_csv.head())

        try:
            df_csv = df_csv[feature_cols]
            preds = model.predict(df_csv)
            df_csv["Prediction"] = preds

            st.write("### 🎯 Prediction Results")
            st.dataframe(df_csv)

            st.download_button(
                "📥 Download Prediction CSV",
                df_csv.to_csv(index=False),
                "credit_score_predictions.csv"
            )

        except Exception as e:
            st.error(f"Column mismatch: {e}")


# =========================================================================================
# =========================================================================================
# 3️⃣ DATA EXPLORER (UPDATED WITH MODEL FEATURES)
# =========================================================================================
elif menu == "Data Explorer":
    st.subheader("📊 Interactive Data Explorer")

    # ------------------------------
    # Show Model Feature Schema
    # ------------------------------
    st.markdown("### 📌 Features Used in the Current ML Model")
    st.info(f"Total Features: **{len(feature_cols)}**")

    st.dataframe(pd.DataFrame({"Feature Name": feature_cols}))

    st.markdown("---")

    # ------------------------------
    # Allow Uploading User Dataset
    # ------------------------------
    uploaded_file = st.file_uploader("📂 Upload Dataset for Exploration (CSV Optional)", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        st.markdown("### 🔍 Data Preview")
        st.dataframe(df.head())

        st.markdown("### 📈 Dataset Statistics")
        st.dataframe(df.describe(), use_container_width=True)

        # --------------------------
        # Column Selector for Plotting
        # --------------------------
        num_cols = df.select_dtypes(include=np.number).columns

        if len(num_cols) >= 2:
            st.markdown("### 📊 Create a Scatter Plot")

            col_x, col_y = st.columns(2)
            x_axis = col_x.selectbox("X-axis", num_cols)
            y_axis = col_y.selectbox("Y-axis", num_cols)

            fig = px.scatter(df, x=x_axis, y=y_axis, title=f"{x_axis} vs {y_axis}")
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Upload a CSV file to explore dataset values & statistics.")

# =========================================================================================
# 4️⃣ FEATURE IMPORTANCE (PREMIUM)
# =========================================================================================
elif menu == "Feature Importance":
    st.subheader("🔥 Feature Importance Visualization")

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        fig = px.bar(
            x=importances,
            y=feature_cols,
            orientation='h',
            title="Feature Importance (Random Forest)",
            color=importances
        )
        st.plotly_chart(fig, use_container_width=True)

    elif hasattr(model, "coef_"):
        coef = model.coef_[0]
        fig = px.bar(
            x=coef,
            y=feature_cols,
            orientation='h',
            title="Feature Weights (Logistic Regression)",
            color=coef
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Feature importance not available for this model.")
