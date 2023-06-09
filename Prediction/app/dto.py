from pydantic import BaseModel


class HealthAnalyzerDTO(BaseModel):
    totalCholesterol: int


class HealthForm(BaseModel):
    primarySecondaryCvrm: str
    hypertension: str
    gender: str
    smokingStatus: str
    organisationName: str
    glucoseFasting: int
    systolicBloodPressure: int
    diastolicBloodPressure: int
    bmi: int
    age: int
    userId: int

    class Config:
        orm_mode = True
