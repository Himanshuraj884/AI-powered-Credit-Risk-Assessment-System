# AI-Powered Credit Risk Assessment System

Project-aligned stack: Python, Flask, Scikit-learn, SHAP, Chart.js, Pandas, NumPy.

## What it does
An end-to-end portfolio credit-risk assessment application. A user enters applicant and loan information, the system engineers financial features, predicts high/low risk with a Random Forest classifier, converts the probability into a lending workflow decision, and explains the prediction with SHAP.

## Features
- Split-screen **Credit Risk AI** applicant form matching the supplied reference design.
- Input validation with useful error messages.
- Shared feature-engineering code used by both training and serving to prevent train/serve drift.
- Random Forest model with preprocessing and categorical handling.
- Per-applicant SHAP feature impacts.
- Flask HTML application and `/api/predict` JSON REST endpoint.
- `/health` health endpoint.
- Dashboard with prediction, loan decision, feature-impact chart, explanation, recommendations and a downloadable/printable decision letter.
- Chart.js visualization with an offline HTML fallback if the CDN is unavailable.
- Automated smoke/API tests.
- Automatic model retraining if the bundled model is missing or cannot be loaded because of a library-version mismatch.

## Data note
The public German Credit dataset does not contain income, credit score, or existing monthly debt-payment data. This portfolio implementation generates those three fields **from non-target fields plus a fixed seeded random process for demonstration purposes. Income and existing monthly debt are expressed in the same monthly currency units used by the web form. These are synthetic demo features, not real bureau measurements. A production system would replace them with validated customer/bureau data.

The real `Risk` column from the dataset is used as the model target. The old shortcut of creating a target from credit amount was removed.

**Job field:** `Job` is served to the model exactly as it means in the source dataset — a skill/employment category (0 = unskilled non-resident, 1 = unskilled resident, 2 = skilled, 3 = highly skilled/management) — and the web form's "Job Category" dropdown asks for that directly. An earlier draft of this form asked for "Employment Status" (employed/self-employed/unemployed/retired) and silently remapped it onto this same 0-3 scale as if the two concepts were equivalent; that conflation has been removed so the form now asks for what the model actually learned from.

## Decision policy
The backend has one source of truth for the **model-risk policy**:

- High-risk probability `< 55%` → APPROVED
- `55%` to `< 75%` → NEEDS REVIEW
- `>= 75%` → REJECTED

These thresholds are an explicit portfolio-demo underwriting policy, not a claim about a real bank's cutoffs. The UI reports the model's high-risk probability and the complementary low-risk probability (`100% − high-risk probability`). The risk label uses the same bands as the model-risk policy.

A separate affordability guardrail can still produce a final REJECTED decision when an applicant is far outside the model's demonstrated financial range. In that case, the dashboard explicitly says Affordability rule and the displayed probability remains the model's original probability; the application never rewrites a model probability just to match the final decision.

## Industry-aligned portfolio design
The project separates the model risk estimate** from the **decision/policy layer. The dashboard reports a risk grade (LOW/MEDIUM/HIGH) from the model probability and a separate recommended action. DTI is calculated from existing monthly debt plus proposed EMI divided by gross monthly income; the model probability is never overwritten by a business rule. The 55% and 75% cutoffs are explicitly configurable demonstration-policy values, not universal banking or regulatory thresholds. Production underwriting would calibrate and validate cutoffs on institution-specific portfolio outcomes and governance requirements.

## Affordability guardrail
The Random Forest is trained on synthetic debt/EMI ratios bounded within a realistic range. Ratios far outside that range (e.g. existing debt many times annual income, or an EMI that exceeds the applicant's entire income) are out-of-distribution for the model and it should not be trusted to score them. A configurable affordability guardrail in `pipeline_utils.affordability_override` controls the final decision for those cases. The values are illustrative portfolio-demo policies, not universal banking standards. The model probability is still calculated for transparency, but it is never rewritten to match the rule-based decision.

\n## Dashboard reliability fix\nThe dashboard passes the prediction result as a structured object to the Jinja template and serializes it exactly once with `tojson`. This prevents the previous blank-dashboard failure caused by double-encoding the result as a JSON string before template serialization. The dashboard also labels the model-risk assessment and final policy decision separately, so an affordability-rule rejection cannot be mistaken for the Random Forest prediction.\n\n## Run locally

### Windows
```bat
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python train_model.py
python tests.py
python app.py
```

### macOS/Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train_model.py
python tests.py
python app.py
```

Open: `http://127.0.0.1:5000`

If the bundled model is incompatible with the installed scikit-learn version, `app.py` automatically retrains it from the included dataset.

## API
`POST /api/predict`

Example JSON:
```json
{
  "age": 35,
  "sex": "male",
  "job": 2,
  "housing": "own",
  "saving": "moderate",
  "checking": "moderate",
  "income": 60000,
  "credit_score": 680,
  "existing_monthly_debt": 5000,
  "amount": 3000,
  "duration": 24,
  "purpose": "car"
}
```

## Testing
Run:
```bash
python tests.py
```

The tests cover fresh page loading, dashboard redirect behavior, health, valid API prediction, decision-policy boundaries, the two interview demo profiles, strict numeric/integer validation, invalid financial input, HTML form prediction, dashboard rendering, the affordability guardrail (extreme debt-to-income, EMI exceeding income), preservation of the original model probability under affordability rejection, and recommendation/label consistency.

## Evaluation
The current seeded, leakage-controlled training run reports:
- Training accuracy: 90.5%
- Held-out test accuracy: 90.0%
- Test ROC-AUC: 0.90.3
- High-risk precision: 0.542
- High-risk recall: 0.633
- High-risk F1: 0.583
- 5-fold CV accuracy: 90.2% ± 2.4%




