from app.create_model import CatBoostRegressor, CatBoostPipeline
import pandas as pd


def predict(X_new: pd.DataFrame):

    model = CatBoostRegressor()
    model.load_model("/Prediction/app/Prediction/best_model.cbm")

    categorical_features = ['Primary / Secondary CVRM', 'Hypertension',
                            'Patient Gender', 'Smoking Status',
                            'Organisation Name (CVRM Treatment)',
                            'Organisation Name (CVRM Treatment).1']
    numerical_features = ['Glucose Fasting', 'Systolic Blood Pressure',
                          'Diastolic Blood Pressure',
                          'BMI', 'Age']

    date_features = ['Treatment Startdate CVRM',
                     'Glucose Fasting, last measurement date',
                     'Systolic Blood Pressure, last measurement date',
                     'BMI, last measurement date',
                     'Date Last Contact CVRM',
                     'Total Cholesterol, last measurement date']

    # Инициализируйте объект пайплайна
    pipeline = CatBoostPipeline(categorical_features=categorical_features,
                                numerical_features=numerical_features,
                                date_features=date_features)

    X_new_processed, y = pipeline.preprocess_data(X_new, target_column='')

    return model.predict(X_new_processed)
