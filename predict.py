import joblib
import pandas as pd
import numpy as np

def load_model(model_path):
    return joblib.load(model_path)

def predict_churn(customer_input_dict, model_path="models/Telco_xgb_pipeline.joblib"):
    """
    Predicts the churn probability and risk category for a given customer.
    
    Args:
        customer_input_dict (dict): A dictionary representing a single customer's data.
        model_path (str): Path to the saved joblib pipeline artifact.
            
    Returns:
        dict: containing 'churn_probability', 'predicted_class', and 'risk_category'.
    """
    pipeline = load_model(model_path)
    
    # Convert input to DataFrame (single row)
    df = pd.DataFrame([customer_input_dict])
    
    # Predict
    prob = pipeline.predict_proba(df)[0, 1]
    pred = pipeline.predict(df)[0]
    
    # Determine risk category
    if prob < 0.3:
        risk = "Low Risk"
    elif prob < 0.7:
        risk = "Medium Risk"
    else:
        risk = "High Risk"
        
    return {
        "churn_probability": float(prob),
        "predicted_class": int(pred),
        "risk_category": risk
    }

if __name__ == "__main__":
    # Example usage:
    # Note: Requires exact columns that the model was trained on.
    # You can import predict_churn from this module in app.
    pass
