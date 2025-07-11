import numpy as np
import pandas as pd
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

class CancerPredictor:
    """Cancer prediction model wrapper"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = ['Age', 'Gender', 'BMI', 'Smoking', 'GeneticRisk', 
                             'PhysicalActivity', 'AlcoholIntake', 'CancerHistory']
        self.feature_importance = None
        self.is_trained = False
        
        # Try to load existing model, if not available, train a new one
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize or train the cancer prediction model"""
        try:
            # For this demo, we'll create a trained model based on the notebook patterns
            # In production, you would load a saved model file
            self._train_demo_model()
            logging.info("Cancer prediction model initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize model: {e}")
            raise
    
    def _train_demo_model(self):
        """Train a demo model with synthetic data matching the notebook patterns"""
        # Generate synthetic training data that matches the patterns from the notebook
        np.random.seed(42)
        n_samples = 1500
        
        # Generate features with realistic distributions
        age = np.random.randint(20, 85, n_samples)
        gender = np.random.choice([0, 1, 2], n_samples, p=[0.45, 0.45, 0.1])
        bmi = np.random.normal(25, 8, n_samples)
        bmi = np.clip(bmi, 15, 45)
        smoking = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
        genetic_risk = np.random.choice([0, 1, 2], n_samples, p=[0.4, 0.4, 0.2])
        physical_activity = np.random.uniform(2, 10, n_samples)
        alcohol_intake = np.random.uniform(0, 5, n_samples)
        cancer_history = np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
        
        # Create more realistic target variable based on risk factors
        risk_score = (
            (age - 20) * 0.02 +  # Age factor
            gender * 0.15 +      # Gender factor
            (bmi - 22) * 0.05 +  # BMI factor
            smoking * 0.3 +      # Smoking factor
            genetic_risk * 0.25 + # Genetic risk factor
            (10 - physical_activity) * 0.1 + # Physical activity (inverted)
            alcohol_intake * 0.1 + # Alcohol factor
            cancer_history * 0.4   # Cancer history factor
        )
        
        # Add some noise and convert to binary
        risk_score += np.random.normal(0, 0.2, n_samples)
        diagnosis = (risk_score > 1.5).astype(int)
        
        # Create DataFrame
        X = pd.DataFrame({
            'Age': age,
            'Gender': gender,
            'BMI': bmi,
            'Smoking': smoking,
            'GeneticRisk': genetic_risk,
            'PhysicalActivity': physical_activity,
            'AlcoholIntake': alcohol_intake,
            'CancerHistory': cancer_history
        })
        
        y = diagnosis
        
        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        # Store feature importance
        self.feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        logging.info(f"Model trained - Train accuracy: {train_score:.3f}, Test accuracy: {test_score:.3f}")
        self.is_trained = True
    
    def predict(self, input_data):
        """Make a cancer prediction"""
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained or loaded")
        
        try:
            # Convert input to DataFrame
            if isinstance(input_data, dict):
                # Map form field names to model feature names
                feature_mapping = {
                    'age': 'Age',
                    'gender': 'Gender', 
                    'bmi': 'BMI',
                    'smoking': 'Smoking',
                    'geneticRisk': 'GeneticRisk',
                    'physicalActivity': 'PhysicalActivity',
                    'alcoholIntake': 'AlcoholIntake',
                    'cancerHistory': 'CancerHistory'
                }
                
                # Create feature vector
                features = []
                for form_field, model_feature in feature_mapping.items():
                    if form_field in input_data:
                        features.append(input_data[form_field])
                    else:
                        raise ValueError(f"Missing required field: {form_field}")
                
                X = np.array(features).reshape(1, -1)
            else:
                X = np.array(input_data).reshape(1, -1)
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Make prediction
            prediction = self.model.predict(X_scaled)[0]
            probability = self.model.predict_proba(X_scaled)[0]
            
            # Calculate risk factors contribution
            risk_factors = self._calculate_risk_factors(input_data)
            
            # Prepare result
            result = {
                'prediction': int(prediction),
                'prediction_text': 'Positive Cancer Risk' if prediction == 1 else 'Negative Cancer Risk',
                'probability': {
                    'no_cancer': float(probability[0]),
                    'cancer': float(probability[1])
                },
                'confidence': float(max(probability)),
                'risk_percentage': float(probability[1] * 100),
                'risk_factors': risk_factors,
                'feature_importance': self.feature_importance,
                'recommendation': self._get_recommendation(prediction, probability[1], risk_factors)
            }
            
            return result
            
        except Exception as e:
            logging.error(f"Prediction error: {e}")
            raise
    
    def _calculate_risk_factors(self, input_data):
        """Calculate individual risk factor contributions"""
        risk_factors = {}
        
        # Age risk
        age = input_data.get('age', 0)
        if age < 30:
            risk_factors['age'] = {'level': 'Low', 'score': 1, 'description': 'Young age is protective'}
        elif age < 50:
            risk_factors['age'] = {'level': 'Moderate', 'score': 2, 'description': 'Middle age carries moderate risk'}
        else:
            risk_factors['age'] = {'level': 'High', 'score': 3, 'description': 'Advanced age increases risk'}
        
        # Gender risk
        gender = input_data.get('gender', 0)
        gender_names = {0: 'Male', 1: 'Female', 2: 'Other'}
        risk_factors['gender'] = {
            'level': 'Moderate' if gender == 1 else 'Moderate', 
            'score': 2, 
            'description': f'{gender_names[gender]} - Gender-specific risks apply'
        }
        
        # BMI risk
        bmi = input_data.get('bmi', 0)
        if bmi < 18.5:
            risk_factors['bmi'] = {'level': 'Moderate', 'score': 2, 'description': 'Underweight may affect immune system'}
        elif bmi < 25:
            risk_factors['bmi'] = {'level': 'Low', 'score': 1, 'description': 'Normal weight is protective'}
        elif bmi < 30:
            risk_factors['bmi'] = {'level': 'Moderate', 'score': 2, 'description': 'Overweight increases risk'}
        else:
            risk_factors['bmi'] = {'level': 'High', 'score': 3, 'description': 'Obesity significantly increases risk'}
        
        # Smoking risk
        smoking = input_data.get('smoking', 0)
        risk_factors['smoking'] = {
            'level': 'High' if smoking == 1 else 'Low',
            'score': 3 if smoking == 1 else 1,
            'description': 'Major cancer risk factor' if smoking == 1 else 'Non-smoking is protective'
        }
        
        # Genetic risk
        genetic_risk = input_data.get('geneticRisk', 0)
        genetic_levels = {0: 'Low', 1: 'Moderate', 2: 'High'}
        risk_factors['genetic'] = {
            'level': genetic_levels[genetic_risk],
            'score': genetic_risk + 1,
            'description': f'{genetic_levels[genetic_risk]} genetic predisposition'
        }
        
        # Physical activity
        physical_activity = input_data.get('physicalActivity', 0)
        if physical_activity >= 7:
            risk_factors['physical_activity'] = {'level': 'Low', 'score': 1, 'description': 'High activity is protective'}
        elif physical_activity >= 4:
            risk_factors['physical_activity'] = {'level': 'Moderate', 'score': 2, 'description': 'Moderate activity'}
        else:
            risk_factors['physical_activity'] = {'level': 'High', 'score': 3, 'description': 'Low activity increases risk'}
        
        # Alcohol intake
        alcohol = input_data.get('alcoholIntake', 0)
        risk_factors['alcohol'] = {
            'level': 'Moderate' if alcohol == 1 else 'Low',
            'score': 2 if alcohol == 1 else 1,
            'description': 'Regular alcohol consumption increases risk' if alcohol == 1 else 'No alcohol consumption is protective'
        }
        
        # Cancer history
        history = input_data.get('cancerHistory', 0)
        risk_factors['cancer_history'] = {
            'level': 'High' if history == 1 else 'Low',
            'score': 3 if history == 1 else 1,
            'description': 'Previous cancer significantly increases risk' if history == 1 else 'No cancer history is protective'
        }
        
        return risk_factors
    
    def _get_recommendation(self, prediction, cancer_probability, risk_factors):
        """Generate personalized recommendations"""
        recommendations = []
        
        if prediction == 1 or cancer_probability > 0.5:
            recommendations.append("⚠️ High risk detected - Consult with an oncologist immediately")
            recommendations.append("📋 Schedule comprehensive cancer screening tests")
        else:
            recommendations.append("✅ Low risk detected - Continue regular health maintenance")
        
        # Risk factor specific recommendations
        high_risk_factors = [factor for factor, data in risk_factors.items() if data['score'] >= 3]
        
        if 'smoking' in high_risk_factors:
            recommendations.append("🚭 Smoking cessation is critical - seek professional help")
        
        if 'bmi' in high_risk_factors:
            recommendations.append("🏃‍♀️ Weight management through diet and exercise")
        
        if 'physical_activity' in high_risk_factors:
            recommendations.append("💪 Increase physical activity to at least 150 minutes/week")
        
        if 'alcohol' in high_risk_factors:
            recommendations.append("🍷 Consider reducing alcohol consumption")
        
        # General recommendations
        recommendations.append("🥗 Maintain a healthy diet rich in fruits and vegetables")
        recommendations.append("😴 Ensure adequate sleep (7-9 hours per night)")
        recommendations.append("🧘‍♀️ Manage stress through relaxation techniques")
        recommendations.append("👨‍⚕️ Regular health check-ups and screenings")
        
        return recommendations
