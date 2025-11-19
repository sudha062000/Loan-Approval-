import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
import joblib

# 1. LOAD TRAIN DATA
train_df = pd.read_csv("trainloan.csv")  # file must be in backend folder

# Convert target 'Y' / 'N' -> 1 / 0
train_df["Loan_Status"] = (train_df["Loan_Status"] == "Y").astype(int)

# 2. FEATURE LISTS (must match your CSV)
numeric_features = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
]

categorical_features = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Credit_History",
    "Property_Area",
]

X = train_df[numeric_features + categorical_features]
y = train_df["Loan_Status"]

# 3. TRAIN / VALID SPLIT
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. PREPROCESSORS (shared by all models)
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median"))]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# 5. DEFINE CANDIDATE MODELS
models = {
    "LogisticRegression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",   # helps if classes are imbalanced
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
    ),
    "GradientBoosting": GradientBoostingClassifier(
        random_state=42
    ),
}

results = []
best_model_name = None
best_pipeline = None
best_score = -1.0  # we'll use ROC-AUC as main metric

print("========== MODEL COMPARISON ==========")
for name, model in models.items():
    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_val)
    y_proba = pipe.predict_proba(X_val)[:, 1]

    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    rec = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    roc = roc_auc_score(y_val, y_proba)

    results.append((name, acc, prec, rec, f1, roc))

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1-score : {f1:.3f}")
    print(f"ROC-AUC  : {roc:.3f}")

    # pick best based on ROC-AUC (you could choose F1 instead)
    if roc > best_score:
        best_score = roc
        best_model_name = name
        best_pipeline = pipe

print("\n======================================")
print("Summary (validation set):")
print("Model              Acc   Prec  Rec   F1    ROC-AUC")
for name, acc, prec, rec, f1, roc in results:
    print(f"{name:16} {acc:0.3f} {prec:0.3f} {rec:0.3f} {f1:0.3f} {roc:0.3f}")

print("\nBest model based on ROC-AUC:", best_model_name)

# 6. DETAILED REPORT FOR BEST MODEL
y_pred_best = best_pipeline.predict(X_val)
print("\nClassification report for BEST model:")
print(classification_report(y_val, y_pred_best))

# 7. SAVE BEST MODEL BUNDLE
bundle = {
    "pipeline": best_pipeline,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "best_model_name": best_model_name,
}
joblib.dump(bundle, "model.joblib")
print(f"\n✅ Saved best model ({best_model_name}) to model.joblib")
