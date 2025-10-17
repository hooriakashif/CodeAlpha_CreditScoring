import pandas as pd
import os
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the dataset
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
df = pd.read_csv(os.path.join(DATA_DIR, "credit.csv"))

# Initial exploration
print("✅ Dataset loaded for preprocessing")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# Assume the last column (A16) is the target ('+' or '-')
# Move it to the first column for clarity and set as target
target_column = df.columns[-1]
df = pd.concat([df[target_column], df.drop(columns=[target_column])], axis=1)

# Handle missing values (fill numerical columns with median)
for column in df.columns:
    if df[column].dtype in ['int64', 'float64']:
        df[column] = df[column].fillna(df[column].median())

# Encode categorical features (e.g., A1, A4, A5, etc.) and target
label_encoders = {}
for column in df.columns:
    if df[column].dtype == 'object':
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

# Target mapping: '+' -> 1, '-' -> 0 (already handled by LabelEncoder if A16 is '+'/'-')
target = df.iloc[:, 0]  # First column after reordering
features = df.iloc[:, 1:]

# Save processed data and encoders
os.makedirs(OUTPUT_DIR, exist_ok=True)
processed_data_path = os.path.join(OUTPUT_DIR, "processed_credit_data.pkl")
joblib.dump({"features": features, "target": target, "encoders": label_encoders}, processed_data_path)
print(f"✅ Processed data saved to {processed_data_path}")

# Basic info
print("Processed Shape (Features, Target):", features.shape, target.shape)
print("Sample of processed data:\n", features.head())