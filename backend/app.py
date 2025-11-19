from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # allow calls from your frontend in dev

# =========================
# LOAD MODEL BUNDLE
# =========================
bundle = joblib.load("model.joblib")

pipeline = bundle["pipeline"]
numeric_features = bundle["numeric_features"]
categorical_features = bundle["categorical_features"]
best_model_name = bundle.get("best_model_name", "Unknown")
selection_metric = bundle.get("selection_metric", "roc_auc")
metrics_by_model = bundle.get("metrics_by_model", {})

print("🚀 Loaded model bundle")
print(f"   → Best model name    : {best_model_name}")
print(f"   → Selection metric   : {selection_metric}")
print(f"   → Numeric features   : {numeric_features}")
print(f"   → Categorical features: {categorical_features}")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/model-info", methods=["GET"])
def model_info():
    """
    Endpoint to inspect which model is in use and its validation metrics.
    """
    return jsonify(
        {
            "best_model_name": best_model_name,
            "selection_metric": selection_metric,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "metrics_by_model": metrics_by_model,
        }
    ), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expect JSON like:
    {
      "ApplicantIncome": 5000,
      "CoapplicantIncome": 2000,
      "LoanAmount": 150,
      "Loan_Amount_Term": 360,
      "Gender": "Male",
      "Married": "Yes",
      "Dependents": "1",
      "Education": "Graduate",
      "Self_Employed": "No",
      "Credit_History": "1.0",
      "Property_Area": "Urban"
    }
    """
    data = request.get_json()

    try:
        row = {}

        # numeric features -> float
        for col in numeric_features:
            val = data.get(col)
            row[col] = float(val) if val not in [None, ""] else None

        # categorical features -> string (can be None)
        for col in categorical_features:
            row[col] = data.get(col)

        # Create single-row DataFrame
        X_input = pd.DataFrame([row])

        # Predict with the loaded pipeline
        pred = pipeline.predict(X_input)[0]
        proba = pipeline.predict_proba(X_input)[0, 1]

        # Debug log
        print("DEBUG row:", row, "→ predicted:", pred, "probability:", proba)

        return jsonify(
            {
                "approved": bool(pred),
                "probability": float(proba),
                "model_used": best_model_name,
            }
        ), 200

    except Exception as e:
        print("Error in /predict:", e)
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
