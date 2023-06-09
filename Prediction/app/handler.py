from datetime import datetime

from fastapi import APIRouter
from app.dto import HealthForm
from app.prediction_model import predict
import pandas as pd

router = APIRouter()


@router.post('/predict')
def predict_router(healthForm: HealthForm):

    date_now = datetime.now()

    data = [healthForm.organisationName, 1, date_now, healthForm.glucoseFasting, date_now,
            date_now, healthForm.systolicBloodPressure, healthForm.diastolicBloodPressure,
            date_now, healthForm.bmi, date_now, healthForm.primarySecondaryCvrm, healthForm.smokingStatus, date_now,
            healthForm.gender, healthForm.age, healthForm.hypertension]

    df = pd.DataFrame([data], columns=['Organisation Name (CVRM Treatment)',
                                     'Organisation Name (CVRM Treatment).1', 'Treatment Startdate CVRM',
                                     'Glucose Fasting', 'Glucose Fasting, last measurement date',
                                     'Total Cholesterol, last measurement date', 'Systolic Blood Pressure',
                                     'Diastolic Blood Pressure', 'Systolic Blood Pressure, last measurement date',
                                     'BMI', 'BMI, last measurement date', 'Primary / Secondary CVRM', 'Smoking Status',
                                     'Date Last Contact CVRM', 'Patient Gender', 'Age', 'Hypertension'])
    res = predict(df)

    return {'totalCholesterol': res[0]}
