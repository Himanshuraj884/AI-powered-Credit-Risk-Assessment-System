"""Automated smoke and API tests. Run: python tests.py"""
from app import app, model, applicant_frame
from pipeline_utils import DECISION_THRESHOLDS, decide_loan, validate_applicant

VALID = {
    "age": 35, "sex": "male", "job": 2, "housing": "own",
    "saving": "moderate", "checking": "moderate", "income": 60000,
    "credit_score": 680, "existing_monthly_debt": 5000, "amount": 3000,
    "duration": 24, "purpose": "car",
}
client = app.test_client()

# Single source of truth for the configurable demo decision policy.
assert DECISION_THRESHOLDS == {"approve_below": 55.0, "reject_at_or_above": 75.0}
assert decide_loan(54.99)[0] == "APPROVED"
assert decide_loan(55.00)[0] == "NEEDS REVIEW"
assert decide_loan(74.99)[0] == "NEEDS REVIEW"
assert decide_loan(75.00)[0] == "REJECTED"

# Fresh pages and health.
r = client.get("/")
assert r.status_code == 200
r = client.get("/dashboard", follow_redirects=False)
assert r.status_code == 302 and r.headers["Location"].endswith("/")
r = client.get("/health")
assert r.status_code == 200 and r.json["model_loaded"] is True

# Valid API prediction and response contract.
r = client.post("/api/predict", json=VALID)
assert r.status_code == 200, r.text
result = r.get_json()
for key in ["prediction", "risk_grade", "risk_probability", "low_risk_probability",
            "decision_label", "decision_reason", "decision_source",
            "features", "shap_features", "recommendations"]:
    assert key in result
assert 0 <= result["risk_probability"] <= 100
assert result["risk_probability"] + result["low_risk_probability"] == 100
assert result["decision_label"] in {"APPROVED", "NEEDS REVIEW", "REJECTED"}
assert result["risk_grade"] in {"LOW", "MEDIUM", "HIGH"}
assert (result["risk_grade"] == "LOW") == (result["prediction"] == "Low Risk")
assert (result["risk_grade"] == "MEDIUM") == (result["prediction"] == "Borderline Risk")
assert (result["risk_grade"] == "HIGH") == (result["prediction"] == "High Risk")

# Deterministic regression cases covering all three model-policy bands.
approved = {
    "age": 41, "sex": "female", "job": 3, "housing": "rent",
    "saving": "rich", "checking": "little", "income": 148183,
    "credit_score": 611, "existing_monthly_debt": 2751, "amount": 245197,
    "duration": 24, "purpose": "domestic appliances",
}
r = client.post("/api/predict", json=approved)
assert r.status_code == 200, r.text
approved_result = r.get_json()
assert approved_result["decision_label"] == "APPROVED", approved_result
assert approved_result["risk_probability"] < DECISION_THRESHOLDS["approve_below"]

review = {
    "age": 28, "sex": "male", "job": 1, "housing": "rent",
    "saving": "rich", "checking": "moderate", "income": 119126,
    "credit_score": 707, "existing_monthly_debt": 7599, "amount": 89811,
    "duration": 60, "purpose": "domestic appliances",
}
r = client.post("/api/predict", json=review)
assert r.status_code == 200, r.text
review_result = r.get_json()
assert review_result["decision_label"] == "NEEDS REVIEW", review_result
assert DECISION_THRESHOLDS["approve_below"] <= review_result["risk_probability"] < DECISION_THRESHOLDS["reject_at_or_above"]

rejected = {
    "age": 65, "sex": "female", "job": 3, "housing": "free",
    "saving": "little", "checking": "moderate", "income": 135826,
    "credit_score": 697, "existing_monthly_debt": 11074, "amount": 76389,
    "duration": 48, "purpose": "car",
}
r = client.post("/api/predict", json=rejected)
assert r.status_code == 200, r.text
rejected_result = r.get_json()
assert rejected_result["decision_label"] == "REJECTED", rejected_result
assert rejected_result["risk_probability"] >= DECISION_THRESHOLDS["reject_at_or_above"]

# The model probability remains the raw classifier probability.
raw_values = validate_applicant(rejected)
raw_X = applicant_frame(rejected, raw_values)
raw_probability = round(float(model.predict_proba(raw_X)[0, 1]) * 100, 2)
assert rejected_result["risk_probability"] == raw_probability

# Validation failures.

# Strict integer validation: API must reject fractional values instead of silently truncating.
for key in ["age", "job", "credit_score", "duration"]:
    bad = dict(VALID); bad[key] = float(bad[key]) + 0.5
    r = client.post("/api/predict", json=bad)
    assert r.status_code == 400, (key, r.text)

for key, value in [
    ("income", 0), ("duration", 0), ("credit_score", 851),
    ("existing_monthly_debt", -1), ("amount", 999),
]:
    bad = dict(VALID); bad[key] = value
    r = client.post("/api/predict", json=bad)
    assert r.status_code == 400, (key, r.text)

bad = dict(VALID); bad["existing_monthly_debt"] = float("nan")
r = client.post("/api/predict", json=bad)
assert r.status_code == 400, r.text

# Affordability rule: DTI > 60% rejects, but the model probability is NOT
# rewritten to make the rule look like a model prediction.
extreme_debt = dict(VALID)
extreme_debt["income"] = 20000
extreme_debt["existing_monthly_debt"] = 20000
r = client.post("/api/predict", json=extreme_debt)
assert r.status_code == 200, r.text
result = r.get_json()
assert result["decision_label"] == "REJECTED", result
assert result["decision_source"] == "Affordability rule", result
raw_values = validate_applicant(extreme_debt)
raw_X = applicant_frame(extreme_debt, raw_values)
raw_probability = round(float(model.predict_proba(raw_X)[0, 1]) * 100, 2)
assert result["risk_probability"] == raw_probability

# Proposed EMI > 50% of income also triggers affordability rejection.
extreme_emi = dict(VALID)
extreme_emi["income"] = 20000
extreme_emi["amount"] = 500000
extreme_emi["duration"] = 6
r = client.post("/api/predict", json=extreme_emi)
assert r.status_code == 200, r.text
assert r.get_json()["decision_label"] == "REJECTED"

# HTML form uses the same field schema as the API.
UI_VALID = {
    "age": 35, "gender": "male", "job": "2", "housing": "own",
    "income": 60000, "credit_score": 680, "existing_monthly_debt": "5000",
    "savings": "low", "account_balance": "low", "amount": 3000,
    "duration": 24, "purpose": "car",
}
r = client.post("/predict", data=UI_VALID)
assert r.status_code == 200, r.text
body = r.get_data(as_text=True)
assert any(label in body for label in ["APPROVED", "NEEDS REVIEW", "REJECTED"])
assert "View Dashboard" in body

r = client.get("/dashboard")
assert r.status_code == 200
dash = r.get_data(as_text=True)
assert "Model high-risk probability" in dash
assert "Risk Grade" in dash
assert "Recommended Action" in dash
assert "Model Risk Assessment" in dash
assert "Final Loan Decision" in dash
assert "model risk grade is LOW" in dash
assert "Low-risk confidence" not in dash
assert "setTimeout(openLetter" not in dash
assert "New Application" in dash
assert "const result = {" in dash
assert "result_json" not in dash

# Recommendation can be triggered from the real numeric debt field.
UI_WITH_DEBT = dict(UI_VALID); UI_WITH_DEBT["existing_monthly_debt"] = "20000"
r = client.post("/predict", data=UI_WITH_DEBT)
assert r.status_code == 200, r.text
assert "Reduce existing debt relative to income." in r.get_data(as_text=True)

print("ALL TESTS PASSED")
