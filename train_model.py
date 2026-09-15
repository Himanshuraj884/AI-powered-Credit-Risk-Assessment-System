"""Train and evaluate the deployed credit-risk pipeline."""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import json
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pipeline_utils import MODEL_COLS, NUMERIC_COLS, CATEGORICAL_COLS, create_bank_features, clean_categoricals

ROOT = Path(__file__).resolve().parent
RANDOM_STATE = 42


def add_demo_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Add portfolio-only fields on the SAME monthly units used by the UI.

    The public German Credit CSV has no income, credit score, or existing-loan
    amount. These demo fields are generated only for this portfolio project
    from non-target applicant attributes plus seeded noise. Income is monthly
    (matching the web form), and Existing Monthly Debt Payment is a monetary amount in the
    same currency units as monthly income.
    """
    df = df.copy()
    rng = np.random.default_rng(RANDOM_STATE)

    age = pd.to_numeric(df["Age"], errors="coerce").fillna(df["Age"].median())
    job = pd.to_numeric(df["Job"], errors="coerce").fillna(1)
    housing_bonus = df["Housing"].map({"own": 12000, "rent": 0, "free": 5000}).fillna(0)
    savings_bonus = df["Saving accounts"].map({
        "little": 0, "moderate": 4000, "quite rich": 8000, "rich": 12000,
    }).fillna(0)

    # Monthly income, deliberately aligned with the UI label/units.
    income = (12000 + age * 650 + job * 12000 + housing_bonus + savings_bonus
              + rng.normal(0, 7000, len(df))).clip(12000, 180000).round(0)
    df["Income"] = income

    checking_bonus = df["Checking account"].map({
        "little": 0, "moderate": 25, "rich": 55,
    }).fillna(0)
    df["Credit Score"] = (500 + age * 2.0 + job * 45 + savings_bonus / 300
                          + checking_bonus + rng.normal(0, 35, len(df))).clip(300, 850).round(0)

    # Existing monthly debt payment is generated on the same monthly units as income.
    # The resulting DTI is the standard debt-obligation-to-gross-income ratio.
    debt_ratio = rng.beta(2.0, 5.0, len(df)) * 0.65
    df["Existing Monthly Debt Payment"] = (income * debt_ratio).clip(0, income * 0.65).round(0)
    return df


def build_pipeline():
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    preprocessor = ColumnTransformer([
        ("num", numeric, NUMERIC_COLS),
        ("cat", categorical, CATEGORICAL_COLS),
    ])
    classifier = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=3,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", classifier)])


def train_and_save():
    df = pd.read_csv(ROOT / "german_credit_data.csv").drop(columns=["Unnamed: 0"], errors="ignore")
    df = clean_categoricals(df)
    if "Risk" not in df.columns:
        raise RuntimeError("Dataset must contain the real Risk target column.")
    y = df.pop("Risk").astype(str).str.lower().map({"good": 0, "bad": 1})
    if y.isna().any(): raise RuntimeError("Risk contains unsupported labels.")
    df = add_demo_fields(df)
    X = create_bank_features(df)[MODEL_COLS]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, stratify=y, random_state=RANDOM_STATE)
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    test_pred = pipeline.predict(X_test)
    test_prob = pipeline.predict_proba(X_test)[:, 1]
    train_acc = accuracy_score(y_train, pipeline.predict(X_train))
    test_acc = accuracy_score(y_test, test_pred)
    auc = roc_auc_score(y_test, test_prob)
    precision = precision_score(y_test, test_pred, zero_division=0)
    recall = recall_score(y_test, test_pred, zero_division=0)
    f1 = f1_score(y_test, test_pred, zero_division=0)
    cm = confusion_matrix(y_test, test_pred).tolist()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(build_pipeline(), X, y, cv=cv, scoring="accuracy", n_jobs=-1)

    print(f"Train accuracy: {train_acc:.3f}")
    print(f"Test accuracy: {test_acc:.3f}")
    print(f"Test ROC-AUC: {auc:.3f}")
    print(f"High-risk precision: {precision:.3f}")
    print(f"High-risk recall: {recall:.3f}")
    print(f"High-risk F1: {f1:.3f}")
    print(f"5-fold CV: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
    print(classification_report(y_test, test_pred, target_names=["Low Risk", "High Risk"]))

    metrics = {
        "algorithm": "Random Forest",
        "train_accuracy": round(float(train_acc), 4),
        "test_accuracy": round(float(test_acc), 4),
        "roc_auc": round(float(auc), 4),
        "precision_high_risk": round(float(precision), 4),
        "recall_high_risk": round(float(recall), 4),
        "f1_high_risk": round(float(f1), 4),
        "confusion_matrix": cm,
        "cv_mean": round(float(cv_scores.mean()), 4),
        "cv_std": round(float(cv_scores.std()), 4),
        "explainability": "SHAP",
        "decision_policy": {"approve_below_pct": 55.0, "manual_review_below_pct": 75.0},
        "policy_note": "Thresholds are configurable portfolio-demo policy values, not regulatory standards.",
    }
    joblib.dump(pipeline, ROOT / "credit_pipeline.pkl")
    (ROOT / "model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Saved credit_pipeline.pkl and model_metrics.json")
    return pipeline, metrics


def main():
    train_and_save()


if __name__ == "__main__":
    main()
