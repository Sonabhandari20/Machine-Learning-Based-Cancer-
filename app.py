import os
import sys
import csv
import uuid
import json
from datetime import datetime

import numpy as np
import joblib
from flask import Flask, render_template, request, flash, redirect, url_for

# Configure Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Needed for flash messages; replace with a secure key

# CSV file path for saving predictions
PREDICTIONS_CSV = 'predictions.csv'

# Load the trained model and scaler
try:
    model = joblib.load('cancer_model.pkl')
    scaler = joblib.load('scaler.pkl')
    app.logger.info("Cancer model and scaler loaded successfully.")
except Exception as e:
    app.logger.error(f"Failed to load model files: {e}")
    model = None
    scaler = None

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if model and scaler are loaded
        if model is None or scaler is None:
            flash('Prediction service is currently unavailable.', 'error')
            return redirect(url_for('index'))

        # Required fields for prediction
        required_fields = [
            'age', 'gender', 'bmi', 'smoking',
            'geneticRisk', 'physicalActivity', 'alcoholIntake', 'cancerHistory'
        ]

        # Check for missing fields in form data
        missing_fields = [field for field in required_fields if not request.form.get(field)]
        if missing_fields:
            flash(f'Missing required fields: {", ".join(missing_fields)}', 'error')
            return redirect(url_for('index'))

        # Parse input data safely
        input_data = {
            'age': float(request.form.get('age')),
            'gender': int(request.form.get('gender')),
            'bmi': float(request.form.get('bmi')),
            'smoking': int(request.form.get('smoking')),
            'geneticRisk': int(request.form.get('geneticRisk')),
            'physicalActivity': float(request.form.get('physicalActivity')),
            'alcoholIntake': int(request.form.get('alcoholIntake')),
            'cancerHistory': int(request.form.get('cancerHistory'))
        }

        # Validate input data values
        validation_errors = []
        if not (0 <= input_data['age'] <= 120):
            validation_errors.append('Age must be between 0 and 120')
        if not (10 <= input_data['bmi'] <= 60):
            validation_errors.append('BMI must be between 10 and 60')
        if input_data['gender'] not in [0, 1, 2]:
            validation_errors.append('Invalid gender value')
        if input_data['smoking'] not in [0, 1]:
            validation_errors.append('Invalid smoking value')
        if input_data['geneticRisk'] not in [0, 1, 2]:
            validation_errors.append('Invalid genetic risk value')
        if input_data['alcoholIntake'] not in [0, 1]:
            validation_errors.append('Invalid alcohol intake value')
        if input_data['cancerHistory'] not in [0, 1]:
            validation_errors.append('Invalid cancer history value')

        if validation_errors:
            for error in validation_errors:
                flash(error, 'error')
            return redirect(url_for('index'))

        # Prepare features and scale
        features = np.array([list(input_data.values())])
        scaled_features = scaler.transform(features)

        # Predict and get probabilities
        prediction = model.predict(scaled_features)
        proba = model.predict_proba(scaled_features)

        risk_percentage = proba[0][1] * 100
        confidence = max(proba[0])
        prediction_text = (
            f"High Risk - {risk_percentage:.1f}% probability"
            if prediction[0] == 1 else
            f"Low Risk - {(proba[0][0] * 100):.1f}% probability"
        )

        prediction_result = {
            'prediction': int(prediction[0]),
            'prediction_text': prediction_text,
            'probability': {
                'no_cancer': float(proba[0][0]),
                'cancer': float(proba[0][1])
            },
            'confidence': float(confidence),
            'risk_percentage': float(risk_percentage),
            'risk_factors': calculate_risk_factors(input_data),
            'feature_importance': get_feature_importance(),
            'recommendation': get_recommendations(prediction[0], risk_percentage, input_data)
        }

        # Save prediction data to CSV file
        save_prediction_to_csv(input_data, prediction_result, request)

        flash(prediction_text, 'success')
        return render_template('result.html', prediction=prediction_result)
    

    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        flash('An error occurred during prediction.', 'error')
        return redirect(url_for('index'))


def save_prediction_to_csv(input_data, prediction_result, request_obj):
    try:
        row_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'session_id': str(uuid.uuid4()),
            'ip_address': request_obj.remote_addr or 'Unknown',
            'user_agent': request_obj.headers.get('User-Agent', '')[:500],
            'age': input_data['age'],
            'gender': input_data['gender'],
            'bmi': input_data['bmi'],
            'smoking': input_data['smoking'],
            'genetic_risk': input_data['geneticRisk'],
            'physical_activity': input_data['physicalActivity'],
            'alcohol_intake': input_data['alcoholIntake'],
            'cancer_history': input_data['cancerHistory'],
            'prediction': prediction_result['prediction'],
            'risk_percentage': prediction_result['risk_percentage'],
            'confidence': prediction_result['confidence'],
            'probability_no_cancer': prediction_result['probability']['no_cancer'],
            'probability_cancer': prediction_result['probability']['cancer'],
            'risk_factors': json.dumps(prediction_result['risk_factors']),
            'feature_importance': json.dumps(prediction_result['feature_importance']),
            'recommendations': json.dumps(prediction_result['recommendation'])
        }

        file_exists = os.path.isfile(PREDICTIONS_CSV)
        with open(PREDICTIONS_CSV, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=row_data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row_data)

        app.logger.info("Prediction saved to CSV.")

    except Exception as e:
        app.logger.error(f"Failed to save prediction: {e}")


def calculate_risk_factors(input_data):
    # TODO: Replace with your actual risk factor calculation logic
    return {"example_factor": "details"}


def get_feature_importance():
    # TODO: Replace with your actual feature importance extraction
    return {"feature1": 0.2, "feature2": 0.3}


def get_recommendations(prediction, risk, input_data):
    # TODO: Replace with your actual recommendations logic
    if prediction == 1:
        return [
            "Consult your healthcare provider.",
            "Eat healthy and balanced meals.",
            "Engage in regular physical activity.",
            "Avoid smoking and excessive alcohol consumption."
        ]
    else:
        return ["Maintain a healthy lifestyle to minimize risk."]


if __name__ == '__main__':
    try:
        app.run(debug=True, port=5000)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
