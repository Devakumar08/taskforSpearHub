# Predictive Analysis for Manufacturing Operations

This project is a simple RESTful API that predicts machine downtime or production defects in manufacturing operations. It allows users to upload a dataset, train a machine learning model, and make predictions about machine downtime based on the provided data.

---

## Features:
- Upload manufacturing data in CSV format.
- Train a machine learning model using Decision Trees.
- Make predictions based on temperature and run time for machine downtime.
- Returns prediction with confidence score.

---

## Prerequisites:
- Python 3.x
- `pip` (Python package installer)
- Flask (for the web framework)
- scikit-learn (for machine learning)
- pandas (for data manipulation)
- joblib (for saving the model)

---

## Installation:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Devakumar08/taskforSpearHub.git
   cd taskforSpearHub
   
### 2. Set Up Virtual Environment
Setting up a virtual environment is optional but highly recommended to isolate your dependencies. Follow these steps:

For Windows:  
- Run the command `python -m venv env` to create a virtual environment.  
- Activate the environment by running `.\env\Scripts\activate`.

For macOS/Linux:  
- Run the command `python3 -m venv env` to create a virtual environment.  
- Activate the environment by running `source env/bin/activate`.

### 3. Install Dependencies
Install the required Python packages by running the following command:  
`pip install -r requirements.txt`.

### 4. Run the Flask Application
Start the Flask server by running the command:  
`python app.py`.

---

## API Endpoints

### 1. **Upload Data**
**Endpoint:** `POST /upload`  
**Description:** Upload a CSV file containing manufacturing data.

#### Request:
- Attach the file in the form field named `file` (e.g., using Postman).

#### Response (Success):
json
{
    "message": "File uploaded successfully"
}

### 2. Train the Model
**Endpoint:** `POST /train`  
**Description:** Train the machine learning model using the uploaded dataset.

#### Request:
- No additional body is required.

#### Response (Success):
json
{
    "message": "Model trained successfully",
    "accuracy": 0.85
}

### 3. Predict Downtime
**Endpoint:** `POST /predict`  
**Description:** Predict whether a machine will experience downtime based on temperature and run time.

#### Request:
Send the following JSON input:
json
{
    "Temperature": 85,
    "Run_Time": 120
}


Response (Success):
{  
    "Downtime": "No",  
    "Confidence": 0.92  
}  
Response (Failure):
{  
    "error": "Invalid input. Provide 'Temperature' and 'Run_Time'."  
}  
