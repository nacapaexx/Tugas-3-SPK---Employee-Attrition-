
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Employee Attrition DSS",
    layout="wide"
)

# =====================================================
# LOAD MODEL & DATASET
# =====================================================

model = joblib.load("model.pkl")

scaler = joblib.load("scaler.pkl")

df = pd.read_csv(
    "WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

# =====================================================
# TITLE
# =====================================================

st.title("Employee Attrition Prediction")

st.markdown("""
Smart Decision Support System for Employee Attrition Prediction

Dashboard ini digunakan untuk:
- memprediksi kemungkinan resign
- menganalisis employee risk
- membantu HR decision making
""")

st.divider()

# =====================================================
# METRIC CARDS
# =====================================================

total_employee = len(df)

attrition_rate = (
    (df["Attrition"] == "Yes").mean() * 100
)

avg_income = df["MonthlyIncome"].mean()

avg_age = df["Age"].mean()

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "👨‍💼 Total Employee",
    total_employee
)

col2.metric(
    "⚠️ Attrition Rate",
    f"{attrition_rate:.2f}%"
)

col3.metric(
    "💰 Avg Income",
    f"${avg_income:,.0f}"
)

col4.metric(
    "🎂 Avg Age",
    f"{avg_age:.0f} Years"
)

st.divider()

# =====================================================
# SIDEBAR FILTER
# =====================================================

st.sidebar.title("📌 Filter Dashboard")

department = st.sidebar.multiselect(
    "Department",
    options=df["Department"].unique(),
    default=df["Department"].unique()
)

filtered_df = df[
    df["Department"].isin(department)
]

# =====================================================
# VISUALIZATION
# =====================================================

st.subheader("📌 Attrition by Department")

fig1 = px.histogram(
    filtered_df,
    x="Department",
    color="Attrition",
    barmode="group",
    template="plotly_dark"
)

st.plotly_chart(
    fig1,
    use_container_width=True
)

st.subheader("📌 Age Distribution")

fig2 = px.histogram(
    filtered_df,
    x="Age",
    nbins=20,
    template="plotly_dark"
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

st.subheader("📌 Overtime vs Attrition")

fig3 = px.histogram(
    filtered_df,
    x="OverTime",
    color="Attrition",
    barmode="group",
    template="plotly_dark"
)

st.plotly_chart(
    fig3,
    use_container_width=True
)

# =====================================================
# DATASET PREVIEW
# =====================================================

st.divider()

st.subheader("📂 Dataset Preview")

st.dataframe(
    filtered_df,
    use_container_width=True
)

# =====================================================
# PREDICTION SECTION
# =====================================================

st.divider()

st.header("🤖 Employee Attrition Prediction")

left, right = st.columns(2)

# ==========================
# LEFT COLUMN
# ==========================

with left:

    age = st.slider(
        "Age",
        18,
        60,
        30
    )

    monthly_income = st.number_input(
        "Monthly Income",
        min_value=1000,
        max_value=20000,
        value=5000
    )

    distance_from_home = st.slider(
        "Distance From Home",
        1,
        30,
        10
    )

# ==========================
# RIGHT COLUMN
# ==========================

with right:

    job_satisfaction = st.selectbox(
        "Job Satisfaction",
        [1, 2, 3, 4]
    )

    overtime = st.selectbox(
        "OverTime",
        ["No", "Yes"]
    )

# =====================================================
# ENCODING
# =====================================================

overtime = 1 if overtime == "Yes" else 0

# =====================================================
# INPUT DATAFRAME
# =====================================================

input_data = pd.DataFrame({
    'Age': [age],
    'MonthlyIncome': [monthly_income],
    'DistanceFromHome': [distance_from_home],
    'JobSatisfaction': [job_satisfaction],
    'OverTime': [overtime]
})

# =====================================================
# SCALING
# =====================================================

input_scaled = scaler.transform(input_data)

# =====================================================
# PREDICTION BUTTON
# =====================================================

if st.button("🔮 Predict Attrition"):

    prediction = model.predict(input_scaled)

    probability = model.predict_proba(input_scaled)

    if prediction[0] == 1:

        st.error(
            "⚠️ Employee is likely to leave the company."
        )

    else:

        st.success(
            "✅ Employee is likely to stay in the company."
        )

    st.metric(
        "Prediction Probability",
        f"{round(np.max(probability)*100,2)}%"
    )

# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "Developed using Streamlit | DSS Employee Attrition"
)

