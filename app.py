import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from model import (
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


st.markdown(
    """
    <style>

    .stApp {
        background-color: #f4f7fb;
    }

    [data-testid="stSidebar"] {
        background-color: #102a43;
    }

    [data-testid="stSidebar"] * {
        color: white;
    }

    .hero {
        background: linear-gradient(
            135deg,
            #102a43,
            #176b87
        );
        padding: 30px;
        border-radius: 20px;
        color: white;
        margin-bottom: 25px;
    }

    .hero h1 {
        font-size: 42px;
        margin-bottom: 5px;
    }

    .hero p {
        font-size: 17px;
    }

    .card {
        background-color: white;
        padding: 22px;
        border-radius: 18px;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
        margin-bottom: 15px;
    }

    .high {
        background-color: #ffe5e5;
        border-left: 7px solid #d62828;
        padding: 25px;
        border-radius: 15px;
    }

    .medium {
        background-color: #fff3d6;
        border-left: 7px solid #f4a261;
        padding: 25px;
        border-radius: 15px;
    }

    .low {
        background-color: #e4f7ed;
        border-left: 7px solid #2a9d68;
        padding: 25px;
        border-radius: 15px;
    }

    .risk-number {
        font-size: 45px;
        font-weight: bold;
    }

    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_data
def load_data():

    data = pd.read_csv(
        "medical_appointments.csv"
    )

    return prepare_data(data)


@st.cache_resource
def create_models():

    data = load_data()

    return train_models(data)


try:

    df = load_data()

    (
        processed_df,
        trained_models,
        results,
        best_model_name,
        best_model
    ) = create_models()

except FileNotFoundError:

    st.error(
        "medical_appointments.csv was not found."
    )

    st.info(
        "Put medical_appointments.csv inside the same folder as app.py."
    )

    st.stop()

except Exception as e:

    st.error(
        "An error occurred while loading the project."
    )

    st.exception(e)

    st.stop()


def age_group(age):

    if age <= 12:
        return "Child"

    if age <= 25:
        return "Young Adult"

    if age <= 45:
        return "Adult"

    if age <= 60:
        return "Middle Age"

    return "Senior"


def risk_reasons(patient):

    reasons = []

    if patient["WaitingDays"] > 14:

        reasons.append(
            "Long waiting period"
        )

    if patient["SMS_received"] == 0:

        reasons.append(
            "SMS reminder not received"
        )

    if patient["Age"] < 25:

        reasons.append(
            "Younger age group"
        )

    if patient["AppointmentWeekday"] >= 5:

        reasons.append(
            "Weekend appointment"
        )

    if patient["Scholarship"] == 1:

        reasons.append(
            "Scholarship indicator present"
        )

    if len(reasons) == 0:

        reasons.append(
            "No major rule-based risk factors detected"
        )

    return reasons


def recommendations(
    patient,
    risk_level
):

    actions = []

    if patient["SMS_received"] == 0:

        actions.append(
            "Send an SMS reminder 24 hours before the appointment."
        )

    if patient["WaitingDays"] > 14:

        actions.append(
            "Send an additional appointment confirmation."
        )

    if risk_level == "HIGH":

        actions.append(
            "Consider a confirmation call."
        )

    elif risk_level == "MEDIUM":

        actions.append(
            "Send an additional reminder."
        )

    else:

        actions.append(
            "Continue the standard reminder process."
        )

    return actions


st.sidebar.markdown(
    "# 🏥 CareAttend AI"
)

st.sidebar.write(
    "Medical Appointment Intelligence"
)

st.sidebar.markdown("---")


page = st.sidebar.radio(
    "Choose Module",
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


st.sidebar.markdown("---")

st.sidebar.success(
    "Best Model: " + best_model_name
)


if page == "🏠 Dashboard":

    st.markdown(
        """
        <div class="hero">

        <h1>🏥 CareAttend AI</h1>

        <p>
        Smart Medical Appointment No-Show Prediction
        </p>

        <p>
        Predict risk • Understand patterns • Take preventive action
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    total = len(processed_df)

    no_show_count = int(
        processed_df["No-show"].sum()
    )

    attended_count = (
        total - no_show_count
    )

    no_show_rate = (
        no_show_count / total * 100
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Total Appointments",
            f"{total:,}"
        )

    with c2:

        st.metric(
            "Attended",
            f"{attended_count:,}"
        )

    with c3:

        st.metric(
            "No-Shows",
            f"{no_show_count:,}"
        )

    with c4:

        st.metric(
            "No-Show Rate",
            f"{no_show_rate:.1f}%"
        )

    st.markdown(
        "## 📈 Appointment Overview"
    )

    col1, col2 = st.columns(2)

    with col1:

        counts = processed_df[
            "No-show"
        ].value_counts()

        attended = counts.get(
            0,
            0
        )

        no_show = counts.get(
            1,
            0
        )

        fig, ax = plt.subplots(
            figsize=(7, 4)
        )

        ax.bar(
            ["Attended", "No-Show"],
            [attended, no_show]
        )

        ax.set_ylabel(
            "Appointments"
        )

        ax.set_title(
            "Appointment Outcome"
        )

        st.pyplot(fig)

        plt.close(fig)

    with col2:

        age_data = processed_df.copy()

        age_data["AgeGroup"] = age_data[
            "Age"
        ].apply(age_group)

        age_rate = (
            age_data
            .groupby("AgeGroup")["No-show"]
            .mean()
            * 100
        )

        order = [
            "Child",
            "Young Adult",
            "Adult",
            "Middle Age",
            "Senior"
        ]

        age_rate = age_rate.reindex(
            order
        ).dropna()

        fig, ax = plt.subplots(
            figsize=(7, 4)
        )

        ax.bar(
            age_rate.index,
            age_rate.values
        )

        ax.set_ylabel(
            "No-Show Percentage"
        )

        ax.set_title(
            "No-Show Rate by Age Group"
        )

        ax.tick_params(
            axis="x",
            rotation=25
        )

        st.pyplot(fig)

        plt.close(fig)

    st.markdown(
        "## ⭐ What makes this project different?"
    )

    a, b, c = st.columns(3)

    with a:

        st.markdown(
            """
            <div class="card">

            <h3>🎯 Risk Score</h3>

            <p>
            Converts prediction probability into
            an easy 0–100 risk score.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    with b:

        st.markdown(
            """
            <div class="card">

            <h3>🔍 Explainable Prediction</h3>

            <p>
            Shows factors associated with
            the predicted risk.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c:

        st.markdown(
            """
            <div class="card">

            <h3>📩 Smart Action</h3>

            <p>
            Suggests operational reminder
            actions based on risk.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )


elif page == "👤 Patient Prediction":

    st.markdown(
        """
        <div class="hero">

        <h1>👤 Patient Risk Assessment</h1>

        <p>
        Enter appointment details to estimate no-show risk.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    left, right = st.columns(2)

    with left:

        st.subheader(
            "👤 Patient Details"
        )

        age = st.number_input(
            "Age",
            min_value=0,
            max_value=120,
            value=30
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

        diabetes = st.selectbox(
            "Diabetes",
            ["No", "Yes"]
        )

    with right:

        st.subheader(
            "📅 Appointment Details"
        )

        alcoholism = st.selectbox(
            "Alcoholism Indicator",
            ["No", "Yes"]
        )

        handicap = st.number_input(
            "Handicap Level",
            min_value=0,
            max_value=4,
            value=0
        )

        sms = st.selectbox(
            "SMS Reminder",
            ["No", "Yes"]
        )

        waiting_days = st.number_input(
            "Waiting Days",
            min_value=0,
            max_value=365,
            value=7
        )

        appointment_day = st.selectbox(
            "Appointment Day",
            [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday"
            ]
        )

        appointment_month = st.selectbox(
            "Appointment Month",
            list(range(1, 13)),
            index=0
        )

    st.markdown("")

    if st.button(
        "🚀 Predict Patient Risk",
        use_container_width=True
    ):

        weekday_map = {
            "Monday": 0,
            "Tuesday": 1,
            "Wednesday": 2,
            "Thursday": 3,
            "Friday": 4,
            "Saturday": 5,
            "Sunday": 6
        }

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
                if sms == "Yes"
                else 0,

            "WaitingDays":
                waiting_days,

            "AppointmentWeekday":
                weekday_map[
                    appointment_day
                ],

            "AppointmentMonth":
                appointment_month
        }

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
                <div class="high">

                <h2>🔴 HIGH RISK</h2>

                <div class="risk-number">
                {risk_score}/100
                </div>

                <p>
                Predicted No-Show Probability:
                <b>{probability:.1%}</b>
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )

        elif risk_level == "MEDIUM":

            st.markdown(
                f"""
                <div class="medium">

                <h2>🟠 MEDIUM RISK</h2>

                <div class="risk-number">
                {risk_score}/100
                </div>

                <p>
                Predicted No-Show Probability:
                <b>{probability:.1%}</b>
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="low">

                <h2>🟢 LOW RISK</h2>

                <div class="risk-number">
                {risk_score}/100
                </div>

                <p>
                Predicted No-Show Probability:
                <b>{probability:.1%}</b>
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )

        st.progress(
            risk_score / 100
        )

        st.markdown(
            "## 🔍 Risk Explanation"
        )

        reasons = risk_reasons(
            patient
        )

        for reason in reasons:

            st.write(
                "• " + reason
            )

        st.markdown(
            "## 📩 Recommended Action"
        )

        actions = recommendations(
            patient,
            risk_level
        )

        for action in actions:

            st.success(
                action
            )

        st.markdown(
            "## 👤 Patient Summary"
        )

        s1, s2, s3, s4 = st.columns(4)

        with s1:

            st.metric(
                "Age Group",
                age_group(age)
            )

        with s2:

            st.metric(
                "Waiting Days",
                waiting_days
            )

        with s3:

            st.metric(
                "SMS",
                sms
            )

        with s4:

            st.metric(
                "Risk",
                risk_level
            )


elif page == "📂 Batch Prediction":

    st.markdown(
        """
        <div class="hero">

        <h1>📂 Batch Prediction</h1>

        <p>
        Upload multiple appointment records and identify high-risk cases.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.info(
        "Upload a CSV with the same columns as medical_appointments.csv."
    )

    uploaded_file = st.file_uploader(
        "Choose CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            upload_df = pd.read_csv(
                uploaded_file
            )

            required = [
                "Gender",
                "ScheduledDay",
                "AppointmentDay",
                "Age",
                "Scholarship",
                "Hipertension",
                "Diabetes",
                "Alcoholism",
                "Handcap",
                "SMS_received"
            ]

            missing = [
                column
                for column in required
                if column not in upload_df.columns
            ]

            if len(missing) > 0:

                st.error(
                    "Missing columns: "
                    + ", ".join(missing)
                )

            else:

                batch_df = prepare_data(
                    upload_df
                )

                probabilities = (
                    best_model
                    .predict_proba(
                        batch_df[
                            [
                                "Age",
                                "Gender",
                                "Scholarship",
                                "Hipertension",
                                "Diabetes",
                                "Alcoholism",
                                "Handcap",
                                "SMS_received",
                                "WaitingDays",
                                "AppointmentWeekday",
                                "AppointmentMonth"
                            ]
                        ]
                    )[:, 1]
                )

                predictions = (
                    probabilities >= 0.5
                ).astype(int)

                batch_df[
                    "Risk Score"
                ] = (
                    probabilities * 100
                ).round().astype(int)

                batch_df[
                    "Predicted No-Show"
                ] = predictions

                batch_df[
                    "Risk Level"
                ] = batch_df[
                    "Risk Score"
                ].apply(
                    lambda x:
                    "HIGH"
                    if x >= 70
                    else
                    "MEDIUM"
                    if x >= 40
                    else
                    "LOW"
                )

                high = int(
                    (
                        batch_df[
                            "Risk Level"
                        ] == "HIGH"
                    ).sum()
                )

                medium = int(
                    (
                        batch_df[
                            "Risk Level"
                        ] == "MEDIUM"
                    ).sum()
                )

                low = int(
                    (
                        batch_df[
                            "Risk Level"
                        ] == "LOW"
                    ).sum()
                )

                c1, c2, c3 = st.columns(3)

                with c1:

                    st.metric(
                        "🔴 High Risk",
                        high
                    )

                with c2:

                    st.metric(
                        "🟠 Medium Risk",
                        medium
                    )

                with c3:

                    st.metric(
                        "🟢 Low Risk",
                        low
                    )

                st.markdown(
                    "## 📋 Prediction Results"
                )

                display_columns = [
                    "Age",
                    "Gender",
                    "WaitingDays",
                    "SMS_received",
                    "Risk Score",
                    "Risk Level",
                    "Predicted No-Show"
                ]

                st.dataframe(
                    batch_df[
                        display_columns
                    ],
                    use_container_width=True
                )

                result_csv = batch_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "⬇️ Download Predictions",
                    data=result_csv,
                    file_name="careattend_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        except Exception as e:

            st.error(
                "Unable to process the uploaded file."
            )

            st.exception(e)


elif page == "📊 Analytics":

    st.markdown(
        """
        <div class="hero">

        <h1>📊 Analytics Studio</h1>

        <p>
        Explore patterns behind appointment attendance.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    analysis_df = processed_df.copy()

    analysis_df["GenderName"] = (
        analysis_df["Gender"]
        .map({
            0: "Female",
            1: "Male"
        })
    )

    analysis_df["SMSName"] = (
        analysis_df["SMS_received"]
        .map({
            0: "No",
            1: "Yes"
        })
    )

    analysis_df["AgeGroup"] = (
        analysis_df["Age"]
        .apply(age_group)
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        gender_filter = st.selectbox(
            "Gender",
            [
                "All",
                "Female",
                "Male"
            ]
        )

    with c2:

        sms_filter = st.selectbox(
            "SMS",
            [
                "All",
                "Yes",
                "No"
            ]
        )

    with c3:

        age_filter = st.selectbox(
            "Age Group",
            [
                "All",
                "Child",
                "Young Adult",
                "Adult",
                "Middle Age",
                "Senior"
            ]
        )

    if gender_filter != "All":

        analysis_df = analysis_df[
            analysis_df["GenderName"]
            == gender_filter
        ]

    if sms_filter != "All":

        analysis_df = analysis_df[
            analysis_df["SMSName"]
            == sms_filter
        ]

    if age_filter != "All":

        analysis_df = analysis_df[
            analysis_df["AgeGroup"]
            == age_filter
        ]

    st.write(
        f"Showing {len(analysis_df):,} appointments"
    )

    col1, col2 = st.columns(2)

    with col1:

        weekday_rate = (
            analysis_df
            .groupby(
                "AppointmentWeekday"
            )["No-show"]
            .mean()
            * 100
        )

        weekday_names = [
            "Mon",
            "Tue",
            "Wed",
            "Thu",
            "Fri",
            "Sat",
            "Sun"
        ]

        x_labels = [
            weekday_names[int(x)]
            for x in weekday_rate.index
        ]

        fig, ax = plt.subplots(
            figsize=(7, 4)
        )

        ax.bar(
            x_labels,
            weekday_rate.values
        )

        ax.set_title(
            "No-Show Rate by Day"
        )

        ax.set_ylabel(
            "No-Show %"
        )

        st.pyplot(fig)

        plt.close(fig)

    with col2:

        sms_rate = (
            analysis_df
            .groupby(
                "SMSName"
            )["No-show"]
            .mean()
            * 100
        )

        fig, ax = plt.subplots(
            figsize=(7, 4)
        )

        ax.bar(
            sms_rate.index,
            sms_rate.values
        )

        ax.set_title(
            "No-Show Rate by SMS"
        )

        ax.set_ylabel(
            "No-Show %"
        )

        st.pyplot(fig)

        plt.close(fig)

    col3, col4 = st.columns(2)

    with col3:

        waiting_df = analysis_df.copy()

        waiting_df["WaitingGroup"] = pd.cut(
            waiting_df["WaitingDays"],
            bins=[
                -1,
                2,
                7,
                14,
                30,
                1000
            ],
            labels=[
                "0-2 Days",
                "3-7 Days",
                "8-14 Days",
                "15-30 Days",
                "30+ Days"
            ]
        )

        waiting_rate = (
            waiting_df
            .groupby(
                "WaitingGroup",
                observed=True
            )["No-show"]
            .mean()
            * 100
        )

        fig, ax = plt.subplots(
            figsize=(7, 4)
        )

        ax.bar(
            waiting_rate.index.astype(str),
            waiting_rate.values
        )

        ax.set_title(
            "No-Show Rate by Waiting Period"
        )

        ax.set_ylabel(
            "No-Show %"
        )

        st.pyplot(fig)

        plt.close(fig)

    with col4:

        condition_names = [
            "Hypertension",
            "Diabetes",
            "Alcoholism"
        ]

        condition_values = [
            int(
                analysis_df[
                    "Hipertension"
                ].sum()
            ),
            int(
                analysis_df[
                    "Diabetes"
                ].sum()
            ),
            int(
                analysis_df[
                    "Alcoholism"
                ].sum()
            )
        ]

        fig, ax = plt.subplots(
            figsize=(7, 4)
        )

        ax.bar(
            condition_names,
            condition_values
        )

        ax.set_title(
            "Patient Health Indicators"
        )

        ax.set_ylabel(
            "Number of Patients"
        )

        st.pyplot(fig)

        plt.close(fig)


elif page == "🧠 Model Lab":

    st.markdown(
        """
        <div class="hero">

        <h1>🧠 Model Laboratory</h1>

        <p>
        Compare machine learning models and understand performance.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    model_names = list(
        results.keys()
    )

    comparison = []

    for name in model_names:

        comparison.append({

            "Model": name,

            "Accuracy":
                results[name]["Accuracy"],

            "Precision":
                results[name]["Precision"],

            "Recall":
                results[name]["Recall"],

            "F1 Score":
                results[name]["F1 Score"]
        })

    comparison_df = pd.DataFrame(
        comparison
    )

    st.dataframe(
        comparison_df,
        use_container_width=True
    )

    st.success(
        "🏆 Best Model: "
        + best_model_name
    )

    best = results[
        best_model_name
    ]

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Accuracy",
            f"{best['Accuracy']:.2%}"
        )

    with c2:

        st.metric(
            "Precision",
            f"{best['Precision']:.2%}"
        )

    with c3:

        st.metric(
            "Recall",
            f"{best['Recall']:.2%}"
        )

    with c4:

        st.metric(
            "F1 Score",
            f"{best['F1 Score']:.2%}"
        )

    st.markdown(
        "## 🔲 Confusion Matrix"
    )

    cm = best[
        "Confusion Matrix"
    ]

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    ax.imshow(cm)

    ax.set_title(
        best_model_name
        + " Confusion Matrix"
    )

    ax.set_xlabel(
        "Predicted"
    )

    ax.set_ylabel(
        "Actual"
    )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels([
        "Attended",
        "No-Show"
    ])

    ax.set_yticklabels([
        "Attended",
        "No-Show"
    ])

    for i in range(2):

        for j in range(2):

            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )

    st.pyplot(fig)

    plt.close(fig)

    st.markdown(
        "## 🔍 Feature Importance"
    )

    importance_df = get_feature_importance(
        best_model
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.barh(
        importance_df["Feature"],
        importance_df["Importance"]
    )

    ax.invert_yaxis()

    ax.set_title(
        "Important Model Features"
    )

    st.pyplot(fig)

    plt.close(fig)


elif page == "🔎 Dataset Explorer":

    st.markdown(
        """
        <div class="hero">

        <h1>🔎 Dataset Explorer</h1>

        <p>
        Explore the appointment dataset used by CareAttend AI.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Rows",
            f"{len(processed_df):,}"
        )

    with c2:

        st.metric(
            "Columns",
            processed_df.shape[1]
        )

    with c3:

        st.metric(
            "Missing Values",
            int(
                processed_df
                .isnull()
                .sum()
                .sum()
            )
        )

    with c4:

        st.metric(
            "Duplicate Rows",
            int(
                processed_df
                .duplicated()
                .sum()
            )
        )

    st.markdown(
        "## 📋 Dataset"
    )

    st.dataframe(
        processed_df,
        use_container_width=True,
        height=450
    )

    st.markdown(
        "## 📊 Statistical Summary"
    )

    st.dataframe(
        processed_df.describe(),
        use_container_width=True
    )

    csv = processed_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download Processed Dataset",
        data=csv,
        file_name="processed_medical_appointments.csv",
        mime="text/csv"
    )


elif page == "ℹ️ About":

    st.markdown(
        """
        <div class="hero">

        <h1>ℹ️ About CareAttend AI</h1>

        <p>
        End-to-end machine learning project for medical appointment attendance prediction.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="card">

        <h2>🎯 Project Objective</h2>

        <p>
        CareAttend AI predicts the probability of a patient missing
        a scheduled medical appointment.
        </p>

        <h3>🤖 Machine Learning Models</h3>

        <ul>
        <li>Logistic Regression</li>
        <li>Random Forest</li>
        <li>Gradient Boosting</li>
        </ul>

        <h3>📊 Evaluation Metrics</h3>

        <ul>
        <li>Accuracy</li>
        <li>Precision</li>
        <li>Recall</li>
        <li>F1 Score</li>
        <li>Confusion Matrix</li>
        </ul>

        <h3>⭐ Unique Features</h3>

        <ul>
        <li>Patient Risk Score from 0 to 100</li>
        <li>Low, Medium and High Risk Categories</li>
        <li>Risk Explanation</li>
        <li>Smart Reminder Recommendations</li>
        <li>Batch Prediction</li>
        <li>Downloadable Predictions</li>
        <li>Analytics Dashboard</li>
        <li>Model Comparison</li>
        <li>Feature Importance</li>
        </ul>

        <h3>👩‍💻 Developer</h3>

        <p>
        Inakshi Hansika S S<br>
        B.Sc. Data Science<br>
        PSGR Krishnammal College for Women
        </p>

        <h3>🛠 Technology</h3>

        <p>
        Python | Pandas | NumPy | Scikit-learn | Matplotlib | Streamlit
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.warning(
        "This is an educational machine learning prototype and "
        "not a clinical decision-making system."
    )