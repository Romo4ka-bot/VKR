from datetime import datetime

from catboost import CatBoostError
from fastapi import APIRouter
from app.dto import HealthForm
from app.prediction_model import predict
import pandas as pd
from app.create_model import CatBoostRegressor

router = APIRouter()


@router.post('/predict')
def predict_router(healthForm: HealthForm):
    date_now = datetime.strptime(datetime.now().strftime('%Y-%m-%d'), '%Y-%m-%d')

    start_date = datetime.strptime(healthForm.startDate, '%Y-%m-%d')
    data = [healthForm.organisationName, 1, start_date, healthForm.glucoseFasting, date_now,
            date_now, healthForm.systolicBloodPressure, healthForm.diastolicBloodPressure,
            date_now, healthForm.bmi, date_now, healthForm.primarySecondaryCvrm, healthForm.smokingStatus, start_date,
            healthForm.gender, healthForm.age, healthForm.hypertension]

    df = pd.DataFrame([data], columns=['Organisation Name (CVRM Treatment)',
                                       'Organisation Name (CVRM Treatment).1', 'Treatment Startdate CVRM',
                                       'Glucose Fasting', 'Glucose Fasting, last measurement date',
                                       'Total Cholesterol, last measurement date', 'Systolic Blood Pressure',
                                       'Diastolic Blood Pressure', 'Systolic Blood Pressure, last measurement date',
                                       'BMI', 'BMI, last measurement date', 'Primary / Secondary CVRM',
                                       'Smoking Status', 'Date Last Contact CVRM', 'Patient Gender', 'Age',
                                       'Hypertension'])
    res = predict(df)

    print(f'predict: {res}')

    return {'totalCholesterol': res[0]}


@router.get('/is-created-model')
def is_created_model():
    try:
        model = CatBoostRegressor()
        model.load_model("/Prediction/app/Prediction/best_model.cbm")
        return {'statusModelCreated': 'True'}
    except CatBoostError:
        return {'statusModelCreated': 'False'}
