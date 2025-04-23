import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("crop_data.csv") 
 # Ensure this file exists

# Check and clean column names
df.columns = df.columns.str.strip()  # Remove extra spaces
print("Dataset Columns:", df.columns)



# Define features
features = ["N", "P", "K", "Temperature", "Humidity", "pH", "Rainfall"]

# Check for missing feature columns
missing_features = [col for col in features if col not in df.columns]
if missing_features:
    raise KeyError(f"Missing feature columns in dataset: {missing_features}")

# **Fix the target column issue**
possible_targets = ["label", "Crop", "Crop_Type", "crop_name"]  # Common names for target column
target = None
for col in possible_targets:
    if col in df.columns:
        target = col
        break

if target is None:
    raise KeyError("Target column (crop label) not found. Check dataset.")

# Extract features and target
X = df[features]
y = df[target]

# Normalize data (optional)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save model and scaler
joblib.dump(model, "crop_modell.pkl")
joblib.dump(scaler, "scaler.pkl")  # Save the scaler



print("Model training complete. Saved as crop_modell.pkl")
