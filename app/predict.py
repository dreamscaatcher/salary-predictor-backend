from pydantic import BaseModel
import pandas as pd
import joblib
import os

class PredictionFeatures(BaseModel):
    experience_level_encoded: float  # EN=0, MI=1, SE=2, EX=3
    company_size_encoded: float      # S=0, M=1, L=2
    employment_type_PT: int          # 0 or 1
    job_title_Data_Engineer: int     # 0 or 1
    job_title_Data_Manager: int      # 0 or 1
    job_title_Data_Scientist: int    # 0 or 1
    job_title_Machine_Learning_Engineer: int  # 0 or 1

def load_model():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'lin_regress.sav')
    return joblib.load(model_path)

def predict_salary(features: PredictionFeatures):
    # Convert features to DataFrame
    input_data = pd.DataFrame([{
        "experience_level_encoded": features.experience_level_encoded,
        "company_size_encoded": features.company_size_encoded,
        "employment_type_PT": features.employment_type_PT,
        "job_title_Data_Engineer": features.job_title_Data_Engineer,
        "job_title_Data_Manager": features.job_title_Data_Manager,
        "job_title_Data_Scientist": features.job_title_Data_Scientist,
        "job_title_Machine_Learning_Engineer": features.job_title_Machine_Learning_Engineer
    }])
    
    # Load model and make prediction
    model = load_model()
    prediction = model.predict(input_data)[0]
    rounded_prediction = round(prediction, 2)
    
    print(f"Debug: Made prediction: {rounded_prediction}")  # Add debug logging
    return rounded_prediction
