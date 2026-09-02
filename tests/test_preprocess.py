import pandas as pd

from src.data.preprocess import clean, fit_transform, load_raw, TARGET_COL


def test_load_raw_has_expected_shape():
    df = load_raw()
    assert len(df) > 7000
    assert "Churn" in df.columns


def test_clean_encodes_target_as_binary():
    df = load_raw()
    cleaned = clean(df)
    assert set(cleaned[TARGET_COL].unique()) <= {0, 1}


def test_clean_handles_blank_total_charges():
    raw = pd.DataFrame([{
        "customerID": "0001-TEST", "gender": "Female", "SeniorCitizen": 0,
        "Partner": "No", "Dependents": "No", "tenure": 0, "PhoneService": "No",
        "MultipleLines": "No phone service", "InternetService": "DSL",
        "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
        "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": "Month-to-month", "PaperlessBilling": "No",
        "PaymentMethod": "Mailed check", "MonthlyCharges": 20.0,
        "TotalCharges": " ", "Churn": "No",
    }])
    cleaned = clean(raw)
    assert cleaned["TotalCharges"].iloc[0] == 0.0


def test_fit_transform_produces_only_numeric_columns():
    df = load_raw()
    cleaned = clean(df)
    encoded, encoders = fit_transform(cleaned)
    assert encoded.drop(columns=[TARGET_COL]).select_dtypes(include="object").empty
    assert len(encoders) > 0
