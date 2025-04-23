import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,classification_report ,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

try:
    df = pd.read_csv("sample1.csv") 

    df.columns = df.columns.str.strip()

    print(" Dataset Loaded Successfully!")
    print(df.head)
    print(df.shape)
    print(df.columns)


    
except FileNotFoundError:
    print(" Error: Dataset file 'crop_data.csv' not found.")
    exit()


features = ["N", "P", "K", "Temperature", "Humidity", "pH", "Rainfall"]
target = "Crop"

missing_columns = [col for col in features + [target] if col not in df.columns]
if missing_columns:
    print(f" Error: Missing columns in dataset: {missing_columns}")
    exit()

X = df[features]
y = df[target]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print(" Model Training Complete!")

joblib.dump(model, "crop_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print(" Model and Scaler Saved!")


try:
    model = joblib.load("crop_model.pkl")
    scaler = joblib.load("scaler.pkl")
    print(" Trained Model Loaded Successfully!")
    y_pred=model.predict(X_test)

    print(model)
    print("Model Parameters: ")
    print(model.get_params())

    print("Accuracy : ",accuracy_score(y_test,y_pred))

    print("\n Classification Report : ")
    print(classification_report(y_test,y_pred))
    print("\n Confusion Matrix : ")
    print(confusion_matrix(y_test,y_pred))
except Exception as e:
    print(f" Error loading model: {e}")
    exit()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

@app.route("/predict-crop", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        print(" Received Data:", data)

        for feature in features:
            if feature not in data:
                return jsonify({"error": f" Missing feature: {feature}"}), 400

        input_data = np.array([[data[feature] for feature in features]])
        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)
        
        print(" Predicted Crop:", prediction[0])
        

        return jsonify({"crop": prediction[0]})

    except Exception as e:
        print(f" Prediction Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003, debug=True)
