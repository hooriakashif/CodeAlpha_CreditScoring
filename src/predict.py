import pandas as pd
import joblib
import os
import numpy as np

# Load the model and scaler
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
model_data = joblib.load(os.path.join(MODELS_DIR, "credit_model.pkl"))
model = model_data["model"]
scaler = model_data["scaler"]

print("✅ Model and scaler loaded successfully!")

# Sample input with feature names (adjust values based on dataset ranges)
sample_input = np.array([[1, 30.0, 2.0, 1, 0, 10, 3, 2.0, 1, 1, 4, 0, 0, 150.0, 500]], 
                       dtype=object)  # Using object to handle mixed types
sample_input_df = pd.DataFrame(sample_input, columns=['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15'])

# Scale the input
sample_input_scaled = scaler.transform(sample_input_df)

# Predict
prediction = model.predict(sample_input_scaled)[0]
probability = model.predict_proba(sample_input_scaled)[0][1] * 100  # Probability of class 1 (approved)

# Output result
result = "approved" if prediction == 1 else "denied"
print(f"Prediction: The credit application is {result} (Probability: {probability:.2f}%)")