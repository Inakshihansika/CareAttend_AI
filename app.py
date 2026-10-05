import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from model import (
    FEATURES,
    prepare_data,
    train_models,
    predict_patient,
    get_feature_importance
)


st.set_page_config(
    page_title="CareAttend AI",
    page_icon="🏥",
    layout="wide"
)


# =========================================================
# CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background-color: #f4f8fb;
}

h1, h2, h3, h4, h5, h6 {
    color: #12344d !important;
}

p, li {
    color: #344b5e !important;
}

.hero {
    background: linear-gradient(
        135deg,
        #0d3553,
        #176b83
    );
    padding: 35px;
    border-radius: 20px;
    margin-bottom: 25px;
}

.hero h1 {
    color: white !important;
    font-size: 40px;
    margin: 0;
}

.hero p {
    color: white !important;
    font-size: 17px;
    margin-top: 8px;
}

.card {
    background-color: white;
    padding: 25px;
    border-radius: 18px;
    border: 1px solid #dce7ee;
    margin-bottom: 20px;
    box-shadow: 0 5px 18px rgba(0,0,0,0.06);
}

.card h2,
.card h3,
.card h4 {
    color: #12344d !important;
}

.card p,
.card li {
    color: #344b5e !important;
}

.section-title {
    color: #12344d !important;
    font-size: 26px;
    font-weight: bold;
    margin-top: 20px;
    margin-bottom: 15px;
}

.developer {
    background-color: white;
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    border: 1px solid #dce7ee;
    margin-top: 25px;
}

.developer h2 {
    color: #12344d !important;
}

.developer p {
    color: #176b83 !important;
}

.high-risk {
    background-color: #ffe8e8;
    border-left: 7px solid #d62828;
    padding: 25px;
    border-radius: 15px;
}

.high-risk h2 {
    color: #b51f1f !important;
}

.high-risk p {
    color: #7d2929 !important;
}

.medium-risk {
    background-color: #fff4dc;
    border-left: 7px solid #e39a16;
    padding: 25px;
    border-radius: 15px;
}

.medium-risk h2 {
    color: #966000 !important;
}

.medium-risk p {
    color: #76520f !important;
}

.low-risk {
    background-color: #e6f7ed;
    border-left: 7px solid #26965d;
    padding: 25px;
    border-radius: 15px;
}

.low-risk h2 {
    color: #176c42 !important;
}

.low-risk p {
    color: #285d43 !important;
}

.footer {
    text-align: center;
    padding: 30px;
    margin-top: 40px;
    color: #617587 !important;
    border-top: 1px solid #dce7ee;
}

.footer strong {
    color: #12344d !important;
}

[data-testid="stMetric"] {
    background-color: white;
    border-radius: 15px;
    padding: 15px;
    border: 1px solid #dce7ee;
}

[data-testid="stMetricLabel"] {
    color: #617587 !important;
}

[data-testid="stMetricValue"] {
    color: #12344d !important;
}

.stButton > button {
    background-color: #176b83;
    color: white !important;
    border-radius: 10px;
    font-weight: bold;
    border: none;
}

.stButton > button:hover {
    background-color: #0d5268;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():
    return pd.read_csv("medical_appointments.csv")


# =========================================================
# TRAIN MODELS
# =========================================================

@st.cache_resource
def create_models():

    df = load_data()

    return train_models(df)


try:

    (
        data,
        trained_models,
        results,
        best_model_name,
        best_model
    ) = create_models()

except Exception as e:

    st.error("There is an error while loading the model.")

    st.code(str(e))

    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div style="text-align:center;">

        <h1 style="color:white !important;">
        🏥 CareAttend AI
        </h1>

        <p style="color:white !important;">
        Medical Appointment Intelligence
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "👤 Patient Prediction",
            "📂 Batch Prediction",
            "📊 Analytics",
            "🧠 Model Lab",
            "🔎 Dataset Explorer",
            "ℹ️ About"
        ]
    )

    st.markdown("---")

    st.markdown(
        f"""
        <div style="
        background:#176b83;
        padding:15px;
        border-radius:12px;
        text-align:center;
        ">

        <p style="color:white !important;">
        Best Model
        </p>

        <h4 style="color:white !important;">
        {best_model_name}
        </h4>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <br>

        <div style="
        background:#12344d;
        padding:15px;
        border-radius:12px;
        text-align:center;
        ">

        <p style="color:#cde8f2 !important;">
        Developed by
        </p>

        <h4 style="color:white !important;">
        Inakshi Hansika S S
        </h4>

        <p style="color:#cde8f2 !important;">
        B.Sc. Data Science
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# DASHBOARD
# =========================================================

if page == "🏠 Dashboard":

    st.markdown(
        """
        <div class="hero">

        <h1>🏥 CareAttend AI</h1>

        <p>
        Intelligent Medical Appointment No-Show Prediction
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    total = len(data)

    no_show = int(
        data["No-show"].sum()
    )

    attended = total - no_show

    no_show_rate = (
        no_show / total
    ) * 100

    st.markdown(
        """
        <div class="section-title">
        📌 Executive Overview
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Appointments",
        f"{total:,}"
    )

    col2.metric(
        "Attended",
        f"{attended:,}"
    )

    col3.metric(
        "No-Shows",
        f"{no_show:,}"
    )

    col4.metric(
        "No-Show Rate",
        f"{no_show_rate:.1f}%"
    )

    st.markdown(
        """
        <div class="section-title">
        🎯 Project Objective
        </div>

        <div class="card">

        <h3>
        Predict • Prioritize • Prevent
        </h3>

        <p>
        CareAttend AI predicts whether a patient is
        likely to miss a scheduled medical appointment.
        </p>

        <p>
        The system uses patient information,
        appointment details, waiting time,
        SMS reminders and health-related factors
        to estimate appointment no-show risk.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            <div class="card">

            <h3>🧠 Machine Learning Models</h3>

            <ul>
            <li>Logistic Regression</li>
            <li>Random Forest</li>
            <li>Gradient Boosting</li>
            </ul>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="card">

            <h3>📈 Evaluation Metrics</h3>

            <ul>
            <li>Accuracy</li>
            <li>Precision</li>
            <li>Recall</li>
            <li>F1 Score</li>
            <li>Confusion Matrix</li>
            </ul>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        """
        <div class="section-title">
        📊 Attendance Distribution
        </div>
        """,
        unsafe_allow_html=True
    )

    fig, ax = plt.subplots()

    ax.bar(
        ["Attended", "No-Show"],
        [attended, no_show]
    )

    ax.set_ylabel(
        "Number of Appointments"
    )

    ax.set_title(
        "Appointment Attendance"
    )

    st.pyplot(fig)


# =========================================================
# PATIENT PREDICTION
# =========================================================

elif page == "👤 Patient Prediction":

    st.markdown(
        """
        <div class="hero">

        <h1>👤 Patient Risk Assessment</h1>

        <p>
        Predict the probability of a patient missing
        their appointment.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-title">
        🧑‍⚕️ Patient Information
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        age = st.number_input(
            "Age",
            min_value=0,
            max_value=120,
            value=35
        )

        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

        scholarship = st.selectbox(
            "Scholarship",
            ["No", "Yes"]
        )

        hypertension = st.selectbox(
            "Hypertension",
            ["No", "Yes"]
        )

    with col2:

        diabetes = st.selectbox(
            "Diabetes",
            ["No", "Yes"]
        )

        alcoholism = st.selectbox(
            "Alcoholism",
            ["No", "Yes"]
        )

        handicap = st.number_input(
            "Handicap",
            min_value=0,
            max_value=4,
            value=0
        )

        sms_received = st.selectbox(
            "SMS Received",
            ["No", "Yes"]
        )

    with col3:

        scheduled_day = st.date_input(
            "Scheduled Date"
        )

        appointment_day = st.date_input(
            "Appointment Date"
        )

        waiting_days = max(
            0,
            (
                pd.Timestamp(
                    appointment_day
                )
                -
                pd.Timestamp(
                    scheduled_day
                )
            ).days
        )

        appointment_weekday = (
            pd.Timestamp(
                appointment_day
            ).dayofweek
        )

        appointment_month = (
            pd.Timestamp(
                appointment_day
            ).month
        )

        st.info(
            f"Waiting Days: {waiting_days}"
        )

    st.markdown("")

    predict = st.button(
        "🔮 Predict Appointment Risk",
        use_container_width=True
    )

    if predict:

        patient = {

            "Age": age,

            "Gender":
                0
                if gender == "Female"
                else 1,

            "Scholarship":
                1
                if scholarship == "Yes"
                else 0,

            "Hipertension":
                1
                if hypertension == "Yes"
                else 0,

            "Diabetes":
                1
                if diabetes == "Yes"
                else 0,

            "Alcoholism":
                1
                if alcoholism == "Yes"
                else 0,

            "Handcap":
                handicap,

            "SMS_received":
                1
                if sms_received == "Yes"
                else 0,

            "WaitingDays":
                waiting_days,

            "AppointmentWeekday":
                appointment_weekday,

            "AppointmentMonth":
                appointment_month
        }

        try:

            (
                prediction,
                probability,
                risk_score,
                risk_level
            ) = predict_patient(
                best_model,
                patient
            )

            st.markdown("---")

            if risk_level == "HIGH":

                st.markdown(
                    f"""
                    <div class="high-risk">

                    <h2>
                    🔴 HIGH RISK
                    </h2>

                    <p>
                    Risk Score:
                    <strong>
                    {risk_score}/100
                    </strong>
                    </p>

                    <p>
                    This patient has a higher predicted
                    probability of missing the appointment.
                    </p>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.warning(
                    "Recommended Action: Send an additional "
                    "SMS reminder or confirmation call."
                )

            elif risk_level == "MEDIUM":

                st.markdown(
                    f"""
                    <div class="medium-risk">

                    <h2>
                    🟠 MEDIUM RISK
                    </h2>

                    <p>
                    Risk Score:
                    <strong>
                    {risk_score}/100
                    </strong>
                    </p>

                    <p>
                    This patient has a moderate predicted
                    no-show risk.
                    </p>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.info(
                    "Recommended Action: Send a reminder "
                    "before the appointment."
                )

            else:

                st.markdown(
                    f"""
                    <div class="low-risk">

                    <h2>
                    🟢 LOW RISK
                    </h2>

                    <p>
                    Risk Score:
                    <strong>
                    {risk_score}/100
                    </strong>
                    </p>

                    <p>
                    This patient has a lower predicted
                    no-show risk.
                    </p>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.success(
                    "Recommended Action: Normal appointment reminder."
                )

            col1, col2 = st.columns(2)

            col1.metric(
                "Risk Score",
                f"{risk_score}/100"
            )

            col2.metric(
                "Probability",
                f"{probability * 100:.1f}%"
            )

        except Exception as e:

            st.error(
                "Prediction failed."
            )

            st.code(
                str(e)
            )


# =========================================================
# BATCH PREDICTION
# =========================================================

elif page == "📂 Batch Prediction":

    st.markdown(
        """
        <div class="hero">

        <h1>📂 Batch Prediction</h1>

        <p>
        Upload multiple appointment records
        and generate predictions.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            batch_data = pd.read_csv(
                uploaded_file
            )

            st.subheader(
                "Uploaded Data"
            )

            st.dataframe(
                batch_data.head(10),
                use_container_width=True
            )

            prepared = prepare_data(
                batch_data
            )

            missing_columns = [
                column
                for column in FEATURES
                if column not in prepared.columns
            ]

            if missing_columns:

                st.error(
                    "Missing required columns:"
                )

                st.write(
                    missing_columns
                )

            else:

                predictions = (
                    best_model.predict(
                        prepared[FEATURES]
                    )
                )

                probabilities = (
                    best_model
                    .predict_proba(
                        prepared[FEATURES]
                    )[:, 1]
                )

                result = batch_data.copy()

                result["Prediction"] = np.where(
                    predictions == 1,
                    "No-Show",
                    "Attend"
                )

                result["Risk Score"] = (
                    probabilities * 100
                ).round(1)

                result["Risk Level"] = pd.cut(
                    probabilities * 100,
                    bins=[
                        -1,
                        39.99,
                        69.99,
                        100
                    ],
                    labels=[
                        "LOW",
                        "MEDIUM",
                        "HIGH"
                    ]
                )

                st.success(
                    "✅ Prediction completed successfully."
                )

                st.dataframe(
                    result,
                    use_container_width=True
                )

                csv = result.to_csv(
                    index=False
                )

                st.download_button(
                    "⬇️ Download Results",
                    csv,
                    "careattend_predictions.csv",
                    "text/csv",
                    use_container_width=True
                )

        except Exception as e:

            st.error(
                "Batch prediction failed."
            )

            st.code(
                str(e)
            )


# =========================================================
# ANALYTICS
# =========================================================

elif page == "📊 Analytics":

    st.markdown(
        """
        <div class="hero">

        <h1>📊 Appointment Analytics</h1>

        <p>
        Explore appointment attendance patterns.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    analysis_data = data.copy()

    col1, col2 = st.columns(2)

    with col1:

        selected_gender = st.multiselect(
            "Gender",
            ["Female", "Male"],
            default=[
                "Female",
                "Male"
            ]
        )

    with col2:

        sms_filter = st.selectbox(
            "SMS Received",
            [
                "All",
                "Yes",
                "No"
            ]
        )

    gender_display = (
        analysis_data["Gender"]
        .map({
            0: "Female",
            1: "Male"
        })
    )

    analysis_data = analysis_data[
        gender_display.isin(
            selected_gender
        )
    ]

    if sms_filter != "All":

        sms_value = (
            1
            if sms_filter == "Yes"
            else 0
        )

        analysis_data = analysis_data[
            analysis_data["SMS_received"]
            == sms_value
        ]

    st.metric(
        "Filtered Appointments",
        f"{len(analysis_data):,}"
    )

    col1, col2 = st.columns(2)

    with col1:

        gender_chart = (
            analysis_data
            .groupby("Gender")["No-show"]
            .mean()
            .reset_index()
        )

        gender_chart["Gender"] = (
            gender_chart["Gender"]
            .map({
                0: "Female",
                1: "Male"
            })
        )

        fig, ax = plt.subplots()

        ax.bar(
            gender_chart["Gender"],
            gender_chart["No-show"] * 100
        )

        ax.set_title(
            "No-Show Rate by Gender"
        )

        ax.set_ylabel(
            "No-Show Rate (%)"
        )

        st.pyplot(fig)

    with col2:

        sms_chart = (
            analysis_data
            .groupby(
                "SMS_received"
            )["No-show"]
            .mean()
            .reset_index()
        )

        sms_chart["SMS_received"] = (
            sms_chart["SMS_received"]
            .map({
                0: "No SMS",
                1: "SMS Received"
            })
        )

        fig, ax = plt.subplots()

        ax.bar(
            sms_chart["SMS_received"],
            sms_chart["No-show"] * 100
        )

        ax.set_title(
            "No-Show Rate by SMS"
        )

        ax.set_ylabel(
            "No-Show Rate (%)"
        )

        st.pyplot(fig)


# =========================================================
# MODEL LAB
# =========================================================

elif page == "🧠 Model Lab":

    st.markdown(
        """
        <div class="hero">

        <h1>🧠 Model Lab</h1>

        <p>
        Compare machine learning model performance.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    rows = []

    for model_name, metrics in results.items():

        rows.append(
            {
                "Model": model_name,
                "Accuracy": metrics["Accuracy"],
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],
                "F1 Score": metrics["F1 Score"]
            }
        )

    model_df = pd.DataFrame(
        rows
    )

    display_df = model_df.copy()

    for column in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]:

        display_df[column] = (
            display_df[column] * 100
        ).round(2)

    st.subheader(
        "📊 Model Performance"
    )

    st.dataframe(
        display_df,
        use_container_width=True
    )

    st.success(
        f"⭐ Best Model: {best_model_name}"
    )

    st.subheader(
        "📈 Model Comparison"
    )

    fig, ax = plt.subplots()

    x = np.arange(
        len(model_df)
    )

    width = 0.18

    ax.bar(
        x - 1.5 * width,
        model_df["Accuracy"],
        width,
        label="Accuracy"
    )

    ax.bar(
        x - 0.5 * width,
        model_df["Precision"],
        width,
        label="Precision"
    )

    ax.bar(
        x + 0.5 * width,
        model_df["Recall"],
        width,
        label="Recall"
    )

    ax.bar(
        x + 1.5 * width,
        model_df["F1 Score"],
        width,
        label="F1 Score"
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        model_df["Model"]
    )

    ax.set_ylim(
        0,
        1
    )

    ax.legend()

    st.pyplot(fig)

    st.subheader(
        "🎯 Confusion Matrix"
    )

    selected_model = st.selectbox(
        "Select Model",
        list(results.keys())
    )

    matrix = results[
        selected_model
    ]["Confusion Matrix"]

    fig, ax = plt.subplots()

    ax.imshow(
        matrix
    )

    ax.set_xlabel(
        "Predicted"
    )

    ax.set_ylabel(
        "Actual"
    )

    ax.set_xticks(
        [0, 1]
    )

    ax.set_yticks(
        [0, 1]
    )

    ax.set_xticklabels(
        [
            "Attend",
            "No-Show"
        ]
    )

    ax.set_yticklabels(
        [
            "Attend",
            "No-Show"
        ]
    )

    for i in range(2):

        for j in range(2):

            ax.text(
                j,
                i,
                str(
                    matrix[i, j]
                ),
                ha="center",
                va="center"
            )

    st.pyplot(fig)

    st.subheader(
        "⭐ Feature Importance"
    )

    importance = get_feature_importance(
        trained_models[
            selected_model
        ]
    )

    st.dataframe(
        importance,
        use_container_width=True
    )


# =========================================================
# DATASET EXPLORER
# =========================================================

elif page == "🔎 Dataset Explorer":

    st.markdown(
        """
        <div class="hero">

        <h1>🔎 Dataset Explorer</h1>

        <p>
        Explore the medical appointment dataset.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Rows",
        f"{len(data):,}"
    )

    col2.metric(
        "Columns",
        len(data.columns)
    )

    col3.metric(
        "Missing Values",
        int(
            data.isna()
            .sum()
            .sum()
        )
    )

    st.subheader(
        "📋 Dataset Preview"
    )

    st.dataframe(
        data,
        use_container_width=True,
        height=450
    )

    information = pd.DataFrame(
        {
            "Column": data.columns,

            "Data Type": [
                str(
                    data[column].dtype
                )
                for column in data.columns
            ],

            "Missing Values": [
                int(
                    data[column]
                    .isna()
                    .sum()
                )
                for column in data.columns
            ],

            "Unique Values": [
                int(
                    data[column]
                    .nunique()
                )
                for column in data.columns
            ]
        }
    )

    st.subheader(
        "📌 Dataset Information"
    )

    st.dataframe(
        information,
        use_container_width=True
    )


# =========================================================
# ABOUT
# =========================================================

elif page == "ℹ️ About":

    st.markdown(
        """
        <div class="hero">

        <h1>ℹ️ About CareAttend AI</h1>

        <p>
        Machine Learning Based Medical Appointment
        No-Show Prediction System
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="card">

        <h2>
        🎯 Project Objective
        </h2>

        <p>
        CareAttend AI is a machine learning application
        designed to predict whether a patient may miss
        a scheduled medical appointment.
        </p>

        <h2>
        🧠 Technologies Used
        </h2>

        <ul>
        <li>Python</li>
        <li>Pandas</li>
        <li>NumPy</li>
        <li>Scikit-learn</li>
        <li>Matplotlib</li>
        <li>Streamlit</li>
        </ul>

        <h2>
        🤖 Machine Learning Models
        </h2>

        <ul>
        <li>Logistic Regression</li>
        <li>Random Forest</li>
        <li>Gradient Boosting</li>
        </ul>

        <h2>
        🚀 Main Features
        </h2>

        <ul>
        <li>Patient risk prediction</li>
        <li>0–100 risk score</li>
        <li>Low / Medium / High risk classification</li>
        <li>Batch CSV prediction</li>
        <li>Interactive analytics</li>
        <li>Model comparison</li>
        <li>Dataset exploration</li>
        </ul>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="developer">

        <div style="font-size:45px;">
        👩‍💻
        </div>

        <h2>
        Inakshi Hansika S S
        </h2>

        <p>
        B.Sc. Data Science
        </p>

        <p>
        Developer & Data Science Project Creator
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="card">

        <h3>
        ⚠️ Disclaimer
        </h3>

        <p>
        This project is developed for educational
        and demonstration purposes. The predictions
        are not clinically validated and should not
        replace professional medical judgement.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">

    🏥 <strong>CareAttend AI</strong>

    <br>

    Medical Appointment No-Show Prediction

    <br><br>

    Developed by
    <strong>
    Inakshi Hansika S S
    </strong>

    • B.Sc. Data Science

    </div>
    """,
    unsafe_allow_html=True
)
