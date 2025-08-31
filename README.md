# Early Cancer Detection using Machine Learning


##  Introduction
Cancer is one of the leading global health challenges, claiming millions of lives every year. Early detection plays a vital role in improving survival rates and treatment outcomes. However, conventional screening methods are often resource-intensive, time-consuming, and prone to human error.  

This project aims to build a **machine learning–based cancer prediction system** that leverages advanced algorithms to provide accurate, interpretable, and accessible risk predictions.

## Project Overview
The system is designed to:
- Predict cancer risk using health-related data.
- Provide binary output (**High Risk / Low Risk**) with probability scores.
- Recommend lifestyle or medical consultation based on risk levels.
- Log user data securely for continuous analysis and healthcare insights.

##  Machine Learning Models Used
- **Logistic Regression**
- **Random Forest**
- **Support Vector Machine (SVM)**
- **XGBoost**

### Techniques Applied
- **SMOTE** → To handle class imbalance.  
- **GridSearchCV** → For hyperparameter tuning.  
- **SHAP** → For model interpretability (preferred over LIME).  

## Web Application
- Secure web-based interface.  
- Users can input personal and health-related data.  
- System outputs:
  - Risk prediction (High/Low)  
  - Probability score  
  - Health recommendations  

## Objectives
- Enhance **early cancer detection**.  
- Reduce **costs and dependency** on manual screening.  
- Provide **accessible healthcare solutions** for underserved communities.  
- Support **global health agenda** of scalable and inclusive technology.  

## Project Structure
E:\FinalProject
│── app.py                # Main Flask application (entry point)
│── cancer_model.pkl      # Saved ML model
│── scaler.pkl            # Data scaler for preprocessing
│── predictions.csv       # Logs of predictions
│── main.ipynb            # Notebook for model training & testing
│── model_loader.py       # Utility to load model and scaler

├── Cleaned_Data
│   ├── Cancer_data_before_cleaning.csv  # Raw dataset
│   └── Cancer_data_after_cleaning.csv   # Cleaned dataset

├── static
│   ├── css
│   │   └── medical.css   # Stylesheet for frontend
│   └── js
│       └── charts.js     # JavaScript for charts/visualizations

└── templates
    ├── base.html         # Base template
    ├── index.html        # Homepage (user input form)
    └── result.html       # Result page (predictions & scores)
