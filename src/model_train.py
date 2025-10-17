import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Load processed data
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
data = joblib.load(os.path.join(MODELS_DIR, "processed_credit_data.pkl"))

features = data["features"]
target = data["target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100)
}

results = {}
for name, model in models.items():
    print(f"🔹 Training {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])
    report = classification_report(y_test, y_pred, output_dict=True)
    
    results[name] = {"accuracy": accuracy, "roc_auc": roc_auc, "report": report}
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("\n")

# Save the best model (highest ROC-AUC)
best_model_name = max(results, key=lambda x: results[x]["roc_auc"])
best_model = models[best_model_name]
joblib.dump({"model": best_model, "scaler": scaler}, os.path.join(MODELS_DIR, "credit_model.pkl"))
print(f"✅ Best model ({best_model_name}) saved as {os.path.join(MODELS_DIR, 'credit_model.pkl')}")

# Summary
print("✅ Training complete! Accuracy summary:")
for name, result in results.items():
    print(f"{name}: {result['accuracy']:.4f}")