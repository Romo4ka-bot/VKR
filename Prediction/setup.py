from setuptools import setup

setup(
    name='prediction-app',
    version='0.0.1',
    author='Roman L',
    author_email='974078rl@gmail.com',
    description='Prediction Service',
    install_requires=[
        'fastapi==0.70.0',
        'uvicorn==0.15.0',
        'SQLAlchemy==1.4.26',
        'requests==2.26.0',
        'pandas==2.0.2',
        'numpy==1.24.3',
        'scikit-learn==1.2.2',
        'catboost==1.2',
        'seaborn==0.12.2',
        'ipython==8.13.2',
        'openpyxl==3.1.2'
    ],
    scripts=['app/main.py']
)
