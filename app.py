"""Credit Risk AI - Flask application and REST API.

Interview-ready portfolio application aligned with the supplied resume:
Python, Flask, Scikit-learn, SHAP and Chart.js.
"""
from pathlib import Path
import json
import os
import secrets
import joblib
import numpy as np
import pandas as pd
import shap
from flask import Flask, jsonify, redirect, render_template, request, session, url_for

from pipeline_utils import (
    DECISION_THRESHOLDS, MODEL_COLS, aggregate_shap_by_feature,
    affordability_override, create_bank_features, decide_loan,
    risk_label_for, validate_applicant,
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "credit_pipeline.pkl"
METRICS_PATH = ROOT / "model_metrics.json"
app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("CREDIT_RISK_SECRET") or secrets.token_hex(32)


def load_model():
    """Load a compatible pipeline; retrain automatically when needed."""
    try:
        candidate = joblib.load(MODEL_PATH)
        if not hasattr(candidate, "named_steps") or not {"preprocessor", "model"}.issubset(candidate.named_steps):
            raise ValueError("Model artifact is not the expected sklearn pipeline.")
        return candidate
    except Exception as exc:
        app.logger.warning("Model artifact could not be loaded (%s); retraining.", exc)
        from train_model import train_and_save
        pipeline, _ = train_and_save()
        return pipeline


model = load_model()
try:
    MODEL_METRICS = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
except Exception:
    MODEL_METRICS = {
        "algorithm": "Random Forest",
        "test_accuracy": None,
        "roc_auc": None,
        "cv_mean": None,
        "cv_std": None,
        "explainability": "SHAP",
        "decision_policy": DECISION_THRESHOLDS,
        "policy_note": "Thresholds are configurable portfolio-demo policy values, not regulatory standards.",
    }
try:
    explainer = shap.TreeExplainer(model.named_steps["model"])
    SHAP_AVAILABLE = True
except Exception as exc:
    app.logger.exception("SHAP explainer could not be initialized: %s", exc)
    explainer = None
    SHAP_AVAILABLE = False


def applicant_frame(data, values):
    row = {
        "Age": values["age"], "Sex": data["sex"], "Job": values["job"],
        "Housing": data["housing"], "Saving accounts": data["saving"],
        "Checking account": data["checking"], "Credit amount": values["amount"],
        "Duration": values["duration"], "Purpose": data["purpose"],
        "Income": values["income"], "Credit Score": values["credit_score"],
        "Existing Monthly Debt Payment": values["existing_monthly_debt"],
    }
    return create_bank_features(pd.DataFrame([row]))[MODEL_COLS]


def real_shap(X):
    if explainer is None:
        return []
    try:
        pre = model.named_steps["preprocessor"]
        transformed = pre.transform(X)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        raw = explainer.shap_values(transformed)
        if isinstance(raw, list):
            row = np.asarray(raw[1][0])
        else:
            arr = np.asarray(raw)
            if arr.ndim == 3:
                row = arr[0, :, 1]
            elif arr.ndim == 2:
                row = arr[0]
            else:
                raise RuntimeError("Unexpected SHAP output shape.")
        pairs = aggregate_shap_by_feature(row, pre.get_feature_names_out())
        return [{"name": name, "value": round(float(value), 5)} for name, value in pairs[:10]]
    except Exception:
        app.logger.exception("SHAP calculation failed for an otherwise valid prediction.")
        return []


RECOMMENDATION_DEBT_TO_INCOME_THRESHOLD = 0.20


def recommendations(data, values):
    tips = []
    if values["credit_score"] < 650:
        tips.append("Improve the credit score before applying for a larger loan.")
    # Compare against a threshold distinct from the UI's yes/no proxy ratio
    # (see normalize_form_data) so this can actually fire from the web form,
    # not just from direct /api/predict calls with a custom ratio.
    if values["existing_monthly_debt"] > values["income"] * RECOMMENDATION_DEBT_TO_INCOME_THRESHOLD:
        tips.append("Reduce existing debt relative to income.")
    if values["amount"] > values["income"] * 6:
        tips.append("Consider a smaller loan amount or a longer repayment period.")
    if data["saving"] == "little":
        tips.append("Build stronger savings before taking on additional credit.")
    return tips or ["No specific input-level recommendation was triggered."]


def normalize_form_data(form):
    """Map the presentation-friendly UI fields to the model's dataset schema."""
    data = form.copy()
    savings_to_dataset = {"low": "little", "moderate": "moderate", "high": "rich"}
    balance_to_dataset = {"low": "little", "moderate": "moderate", "high": "rich"}

    # Job comes directly from the dataset-aligned 0-3 Job Category field.
    data["job"] = data.get("job", 1)
    data["sex"] = data.get("gender", "male")
    data["housing"] = data.get("housing", "own")
    data["saving"] = savings_to_dataset.get(data.get("savings", "moderate"), "moderate")
    data["checking"] = balance_to_dataset.get(data.get("account_balance", "moderate"), "moderate")

    try:
        income = float(data.get("income", 0) or 0)
    except (TypeError, ValueError):
        income = 0
    # The UI collects actual existing monthly debt payment, so DTI can be
    # calculated using the standard monthly debt-obligation definition.
    try:
        data["existing_monthly_debt"] = float(data.get("existing_monthly_debt", 0) or 0)
    except (TypeError, ValueError):
        data["existing_monthly_debt"] = data.get("existing_monthly_debt", 0)

    for key in ["gender", "savings", "account_balance"]:
        data.pop(key, None)
    return data


def predict_payload(data):
    values = validate_applicant(data)
    X = applicant_frame(data, values)
    probability = float(model.predict_proba(X)[0, 1]) * 100
    predicted_class = int(model.predict(X)[0])
    dti = float(X["DTI"].iloc[0])
    emi_to_income = float(X["EMI_to_Income"].iloc[0])

    # Hard affordability check runs before we trust the model: the ratios here
    # can be pushed far outside anything the model was trained on, and a
    # Random Forest doesn't extrapolate safely past its training distribution.
    override_reason = affordability_override(dti, emi_to_income)
    if override_reason:
        decision_label, decision_color, decision_icon = "REJECTED", "#dc2626", "❌"
        decision_reason = override_reason
        decision_source = "Affordability rule"
    else:
        decision_label, decision_color, decision_icon = decide_loan(probability)
        decision_reason = {
            "APPROVED": "Model risk is below the approval threshold.",
            "NEEDS REVIEW": "Model risk falls in the manual-review band.",
            "REJECTED": "Model risk meets or exceeds the rejection threshold.",
        }[decision_label]
        decision_source = "Model risk policy"

    tips = recommendations(data, values)
    if override_reason:
        # Drop the generic tip that overlaps with the specific override reason
        # so the same point isn't made twice in the list.
        redundant_tip = {
            "The proposed monthly installment exceeds the applicant's income.":
                "Consider a smaller loan amount or a longer repayment period.",
            "Existing debt is too high relative to income to safely extend more credit.":
                "Reduce existing debt relative to income.",
        }.get(override_reason)
        tips = [t for t in tips if t != redundant_tip]
        tips = [override_reason] + tips

    return {
        # The probability is ALWAYS the actual model probability. Business
        # rules can change the final decision, but never rewrite the model score.
        "prediction": risk_label_for(probability),
        "risk_grade": {"Low Risk": "LOW", "Borderline Risk": "MEDIUM", "High Risk": "HIGH"}[risk_label_for(probability)],
        "prediction_value": predicted_class,
        "risk_probability": round(probability, 2),
        "low_risk_probability": round(100 - probability, 2),
        "decision_label": decision_label,
        "decision_color": decision_color,
        "decision_icon": decision_icon,
        "decision_reason": decision_reason,
        "decision_source": decision_source,
        "features": {
            "Monthly Income": round(values["income"], 2),
            "Credit Score": values["credit_score"],
            "Credit Amount": round(values["amount"], 2),
            "Existing Monthly Debt": round(values["existing_monthly_debt"], 2),
            "DTI": round(dti, 2),
            "EMI": round(float(X["EMI"].iloc[0]), 2),
            "EMI_to_Income": round(emi_to_income, 2),
        },
        "shap_features": real_shap(X),
        "recommendations": tips,
        "model_metrics": MODEL_METRICS,
        "policy": {"approve_below_pct": DECISION_THRESHOLDS["approve_below"], "manual_review_below_pct": DECISION_THRESHOLDS["reject_at_or_above"], "note": "Configurable portfolio-demo policy; not a regulatory banking standard."},
    }


@app.get("/")
def home():
    return render_template("index.html", result=session.get("last_result"), form_data={})


@app.post("/predict")
def predict_form():
    try:
        result = predict_payload(normalize_form_data(request.form.to_dict()))
        session["last_result"] = result
        return render_template("index.html", result=result, form_data=request.form.to_dict())
    except ValueError as exc:
        return render_template("index.html", error=str(exc), form_data=request.form.to_dict()), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return render_template("index.html", error="Unable to process the application.", form_data=request.form.to_dict()), 500


@app.post("/api/predict")
def api_predict():
    try:
        result = predict_payload(request.get_json(silent=True) or {})
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        app.logger.exception("API prediction failed")
        return jsonify({"error": "Unable to process the application."}), 500


@app.get("/dashboard")
def dashboard():
    result = session.get("last_result")
    if not result:
        return redirect(url_for("home"))
    return render_template("dashboard.html", result=result)


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None, "shap_available": SHAP_AVAILABLE})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
