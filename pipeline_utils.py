"""Shared feature engineering and business decision logic.

The training script and Flask application import this module so that the
training-serving feature schema cannot drift apart.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

NUMERIC_COLS = [
    "Age", "Job", "Credit amount", "Duration", "DTI", "EMI",
    "EMI_to_Income", "Credit Score Risk", "Income Stability",
    "Savings Score", "Income", "Credit Score", "Existing Monthly Debt Payment",
]

CATEGORICAL_COLS = [
    "Sex", "Housing", "Saving accounts", "Checking account", "Purpose",
]

MODEL_COLS = NUMERIC_COLS + CATEGORICAL_COLS
ASSUMED_ANNUAL_INTEREST_RATE = 12.0
DECISION_THRESHOLDS = {"approve_below": 55.0, "reject_at_or_above": 75.0}

# Affordability guardrails. The model is trained on synthetic demo ratios that
# never exceed these magnitudes, so ratios beyond them are out-of-distribution
# for the classifier and it should not be trusted to score them sensibly.
# A real bank would hard-reject on affordability grounds before ever consulting
# a model, so we do the same here instead of letting the model extrapolate.
MAX_SANE_DTI_PCT = 60.0          # conservative portfolio-demo affordability guardrail
MAX_SANE_EMI_TO_INCOME_PCT = 50.0  # proposed payment should not exceed half of gross monthly income

DATASET_PURPOSES = [
    "business", "car", "domestic appliances", "education",
    "furniture/equipment", "radio/TV", "repairs", "vacation/others",
]
DATASET_HOUSING = ["free", "own", "rent"]
DATASET_SAVINGS = ["little", "moderate", "quite rich", "rich", "none"]
DATASET_CHECKING = ["little", "moderate", "rich", "none"]
DATASET_SEX = ["male", "female"]


def clean_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("none").astype(str)
    return df


def calculate_emi(principal: float, duration_months: float) -> float:
    if principal <= 0 or duration_months <= 0:
        raise ValueError("Credit amount and duration must be greater than zero.")
    monthly_rate = ASSUMED_ANNUAL_INTEREST_RATE / 12 / 100
    n = float(duration_months)
    factor = (1 + monthly_rate) ** n
    return float(principal * monthly_rate * factor / (factor - 1))


def create_bank_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the exact engineered columns used by training and inference."""
    df = clean_categoricals(df.copy())

    income = pd.to_numeric(df["Income"], errors="coerce")
    existing_loans = pd.to_numeric(df["Existing Monthly Debt Payment"], errors="coerce")
    amount = pd.to_numeric(df["Credit amount"], errors="coerce")
    duration = pd.to_numeric(df["Duration"], errors="coerce")
    credit_score = pd.to_numeric(df["Credit Score"], errors="coerce")
    job = pd.to_numeric(df["Job"], errors="coerce")

    # DTI is the standard monthly debt-obligation-to-gross-income ratio.
    # It includes existing monthly debt payments plus the proposed EMI.
    df["EMI"] = [calculate_emi(p, n) for p, n in zip(amount, duration)]
    df["EMI_to_Income"] = (df["EMI"] / income).replace([np.inf, -np.inf], np.nan) * 100
    df["DTI"] = ((existing_loans + df["EMI"]) / income).replace([np.inf, -np.inf], np.nan) * 100
    df["Credit Score Risk"] = ((900 - credit_score) / 600).clip(0, 1)
    df["Income Stability"] = (job / 3).clip(0, 1)
    df["Savings Score"] = df["Saving accounts"].map({
        "none": 0, "little": 1, "moderate": 2, "quite rich": 3, "rich": 4,
    }).fillna(0)

    return df


def decide_loan(risk_probability_pct: float) -> tuple[str, str, str]:
    p = float(risk_probability_pct)
    if not 0 <= p <= 100:
        raise ValueError("Risk probability must be between 0 and 100.")
    if p < DECISION_THRESHOLDS["approve_below"]:
        return "APPROVED", "#16a34a", "✅"
    if p < DECISION_THRESHOLDS["reject_at_or_above"]:
        return "NEEDS REVIEW", "#f59e0b", "⚠️"
    return "REJECTED", "#dc2626", "❌"


def risk_label_for(risk_probability_pct: float) -> str:
    """User-facing risk wording, aligned to the SAME bands as decide_loan so the
    prediction text and the decision badge can never contradict each other."""
    p = float(risk_probability_pct)
    if p < DECISION_THRESHOLDS["approve_below"]:
        return "Low Risk"
    if p < DECISION_THRESHOLDS["reject_at_or_above"]:
        return "Borderline Risk"
    return "High Risk"


def affordability_override(dti_pct: float, emi_to_income_pct: float):
    """Hard affordability rule applied before trusting the model's score.

    The classifier is trained on synthetic demo ratios. Ratios beyond the
    configurable guardrails below are treated as outside the model's demonstrated
    range, so the application applies an explicit affordability policy rather than
    pretending the model probability alone is sufficient. These are illustrative
    portfolio-demo policies, not universal banking rules.
    Returns a reason string, or None if no override applies.
    """
    if emi_to_income_pct is not None and emi_to_income_pct > MAX_SANE_EMI_TO_INCOME_PCT:
        return "The proposed monthly installment exceeds the applicant's income."
    if dti_pct is not None and dti_pct > MAX_SANE_DTI_PCT:
        return "Existing debt is too high relative to income to safely extend more credit."
    return None


def aggregate_shap_by_feature(shap_row, encoded_feature_names):
    totals = {col: 0.0 for col in MODEL_COLS}
    for name, value in zip(encoded_feature_names, shap_row):
        if name.startswith("num__"):
            col = name[5:]
        elif name.startswith("cat__"):
            rest = name[5:]
            col = next((c for c in CATEGORICAL_COLS if rest == c or rest.startswith(c + "_")), None)
        else:
            col = None
        if col in totals:
            totals[col] += float(value)
    return sorted(totals.items(), key=lambda item: abs(item[1]), reverse=True)


def validate_applicant(data: dict) -> dict:
    required = [
        "age", "sex", "job", "housing", "saving", "checking", "income",
        "credit_score", "existing_monthly_debt", "amount", "duration", "purpose",
    ]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError("Missing required fields: " + ", ".join(missing))

    def number(key, kind=float):
        try:
            raw = data[key]
            if isinstance(raw, bool):
                raise ValueError
            value = kind(raw)
            if kind is int and float(raw) != value:
                raise ValueError
            return value
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"{key} must be a valid {'integer' if kind is int else 'number'}.")

    values = {
        "age": number("age", int), "job": number("job", int),
        "income": number("income"), "credit_score": number("credit_score", int),
        "existing_monthly_debt": number("existing_monthly_debt"),
        "amount": number("amount"), "duration": number("duration", int),
    }
    if not all(math.isfinite(float(v)) for v in values.values()):
        raise ValueError("Numeric fields must contain finite values.")
    if not 18 <= values["age"] <= 100: raise ValueError("Age must be between 18 and 100.")
    if not 0 <= values["job"] <= 3: raise ValueError("Job category must be between 0 and 3.")
    if values["income"] <= 0: raise ValueError("Income must be greater than 0.")
    if not 300 <= values["credit_score"] <= 850: raise ValueError("Credit score must be between 300 and 850.")
    if values["existing_monthly_debt"] < 0: raise ValueError("Existing monthly debt payment cannot be negative.")
    if values["income"] < 1000: raise ValueError("Monthly income must be at least ₹1,000.")
    if values["amount"] <= 0: raise ValueError("Credit amount must be greater than 0.")
    if not 1 <= values["duration"] <= 360: raise ValueError("Loan duration must be between 1 and 360 months.")
    if data["sex"] not in DATASET_SEX: raise ValueError("Invalid sex category.")
    if data["housing"] not in DATASET_HOUSING: raise ValueError("Invalid housing category.")
    if data["saving"] not in DATASET_SAVINGS: raise ValueError("Invalid savings category.")
    if data["checking"] not in DATASET_CHECKING: raise ValueError("Invalid checking-account category.")
    if data["purpose"] not in DATASET_PURPOSES: raise ValueError("Invalid loan purpose.")
    return values
