import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


FEATURES = [
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


def convert_binary_column(series):

    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(
            series,
            errors="coerce"
        )

    values = (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )

    mapping = {
        "yes": 1,
        "no": 0,
        "true": 1,
        "false": 0,
        "y": 1,
        "n": 0,
        "1": 1,
        "0": 0
    }

    return values.map(mapping)


def prepare_data(df):

    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
    )

    if "Gender" in df.columns:

        if not pd.api.types.is_numeric_dtype(
            df["Gender"]
        ):

            gender_values = (
                df["Gender"]
                .astype(str)
                .str.strip()
                .str.upper()
            )

            df["Gender"] = gender_values.map({
                "F": 0,
                "M": 1,
                "FEMALE": 0,
                "MALE": 1
            })

        else:

            df["Gender"] = pd.to_numeric(
                df["Gender"],
                errors="coerce"
            )

    if "No-show" in df.columns:

        df["No-show"] = convert_binary_column(
            df["No-show"]
        )

    binary_columns = [
        "Scholarship",
        "Hipertension",
        "Diabetes",
        "Alcoholism",
        "SMS_received"
    ]

    for column in binary_columns:

        if column in df.columns:

            df[column] = convert_binary_column(
                df[column]
            )

    if "Handcap" in df.columns:

        df["Handcap"] = pd.to_numeric(
            df["Handcap"],
            errors="coerce"
        )

    if "Age" in df.columns:

        df["Age"] = pd.to_numeric(
            df["Age"],
            errors="coerce"
        )

    if "ScheduledDay" in df.columns:

        df["ScheduledDay"] = pd.to_datetime(
            df["ScheduledDay"],
            errors="coerce"
        )

    if "AppointmentDay" in df.columns:

        df["AppointmentDay"] = pd.to_datetime(
            df["AppointmentDay"],
            errors="coerce"
        )

    if (
        "ScheduledDay" in df.columns
        and
        "AppointmentDay" in df.columns
    ):

        df["WaitingDays"] = (
            df["AppointmentDay"]
            - df["ScheduledDay"]
        ).dt.days

        df["WaitingDays"] = (
            df["WaitingDays"]
            .fillna(0)
            .clip(lower=0)
        )

        df["AppointmentWeekday"] = (
            df["AppointmentDay"]
            .dt.dayofweek
        )

        df["AppointmentMonth"] = (
            df["AppointmentDay"]
            .dt.month
        )

    numeric_columns = [
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

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

            df[column] = (
                df[column]
                .fillna(0)
            )

    return df


def train_models(df):

    df = prepare_data(df)

    df = df.dropna(
        subset=["No-show"]
    )

    X = df[FEATURES].copy()

    y = df["No-show"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    models = {

        "Logistic Regression": Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced"
                )
            )
        ]),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced"
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
    }

    trained_models = {}

    results = {}

    for name, model in models.items():

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
        )

        results[name] = {

            "Accuracy": accuracy_score(
                y_test,
                predictions
            ),

            "Precision": precision_score(
                y_test,
                predictions,
                zero_division=0
            ),

            "Recall": recall_score(
                y_test,
                predictions,
                zero_division=0
            ),

            "F1 Score": f1_score(
                y_test,
                predictions,
                zero_division=0
            ),

            "Confusion Matrix": confusion_matrix(
                y_test,
                predictions
            )
        }

        trained_models[name] = model

    best_model_name = max(
        results,
        key=lambda name:
        results[name]["F1 Score"]
    )

    best_model = trained_models[
        best_model_name
    ]

    return (
        df,
        trained_models,
        results,
        best_model_name,
        best_model
    )


def predict_patient(
    model,
    patient
):

    patient_df = pd.DataFrame(
        [patient]
    )

    patient_df = patient_df[
        FEATURES
    ]

    prediction = model.predict(
        patient_df
    )[0]

    probability = model.predict_proba(
        patient_df
    )[0][1]

    risk_score = int(
        round(
            probability * 100
        )
    )

    if risk_score >= 70:

        risk_level = "HIGH"

    elif risk_score >= 40:

        risk_level = "MEDIUM"

    else:

        risk_level = "LOW"

    return (
        int(prediction),
        float(probability),
        risk_score,
        risk_level
    )


def get_feature_importance(model):

    if hasattr(
        model,
        "feature_importances_"
    ):

        importance = (
            model.feature_importances_
        )

    elif hasattr(
        model,
        "named_steps"
    ):

        final_model = (
            model.named_steps["model"]
        )

        if hasattr(
            final_model,
            "coef_"
        ):

            importance = np.abs(
                final_model.coef_[0]
            )

        else:

            importance = np.zeros(
                len(FEATURES)
            )

    else:

        importance = np.zeros(
            len(FEATURES)
        )

    result = pd.DataFrame({

        "Feature": FEATURES,

        "Importance": importance

    })

    return result.sort_values(
        "Importance",
        ascending=False
    )