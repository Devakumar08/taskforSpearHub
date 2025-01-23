import pandas as pd
from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)

# Global variables
model = None
uploaded_data = None
scaler = None

# Upload endpoint
@app.route('/upload', methods=['POST'])
def upload_data():
    global uploaded_data
    file = request.files.get('file')
    if not file:
        return {"error": "No file uploaded"}, 400
    uploaded_data = pd.read_csv(file)
    return {"message": "File uploaded successfully"}, 200

# Train endpoint
@app.route('/train', methods=['POST'])
def train_model():
    global model, uploaded_data, scaler
    if uploaded_data is None:
        return {"error": "No data uploaded. Use /upload first."}, 400
    
    # Select features and target
    X = uploaded_data[["Temperature", "Run_Time"]]
    y = uploaded_data["Downtime_Flag"]

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model with hyperparameter tuning
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10]
    }
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='accuracy', cv=3)
    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # Save model and scaler
    joblib.dump(model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    
    return {
        "message": "Model trained successfully",
        "accuracy": round(accuracy, 2),
        "f1_score": round(f1, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2)
    }, 200

# Predict endpoint
@app.route('/predict', methods=['POST'])
def predict():
    global model, scaler
    if model is None or scaler is None:
        return {"error": "Model not trained. Use /train first."}, 400
    
    input_data = request.json
    temperature = input_data.get("Temperature")
    run_time = input_data.get("Run_Time")
    
    if temperature is None or run_time is None:
        return {"error": "Invalid input. Provide 'Temperature' and 'Run_Time'."}, 400
    
    # Scale input
    scaled_input = scaler.transform([[temperature, run_time]])
    
    # Make prediction
    prediction = model.predict(scaled_input)
    confidence = max(model.predict_proba(scaled_input)[0])
    
    return {
        "Downtime": "Yes" if prediction[0] == 1 else "No",
        "Confidence": round(confidence, 2)
    }, 200

if __name__ == "__main__":
    app.run(debug=True)
