import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Create output directory at the start
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "eda_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load processed data
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
data = joblib.load(os.path.join(MODELS_DIR, "processed_credit_data.pkl"))

features = data["features"]
target = data["target"]

df = pd.concat([features, pd.Series(target, name="target")], axis=1)

print("✅ Dataset loaded successfully for EDA")
print("Shape of data:", df.shape)

# Handle missing values (fill with median for numerical columns)
for column in df.columns:
    if df[column].dtype in ['int64', 'float64']:
        df[column] = df[column].fillna(df[column].median())  # Avoid inplace warning

print("\n--- First 5 rows ---")
print(df.head())

print("\n--- Data Information ---")
print(df.info())

print("\n--- Summary Statistics ---")
print(df.describe())

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Target Distribution ---")
print(df['target'].value_counts())

# Visualize target distribution
plt.figure(figsize=(8, 5))
sns.countplot(x='target', data=df)
plt.title('Credit Approval Distribution')
plt.ylabel('Count')
plt.savefig(os.path.join(OUTPUT_DIR, "target_distribution.png"))
plt.close()

# Correlation heatmap (first 10 features)
plt.figure(figsize=(12, 8))
correlation_matrix = df.iloc[:, :10].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Heatmap')
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"))
plt.close()

print("\n✅ EDA Complete!")