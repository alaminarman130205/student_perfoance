from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
import pandas as pd
import joblib

model = joblib.load("student_performance_model.joblib")

app = FastAPI(
    title="Student Performance Prediction API",
    description="Predicts math, reading, and writing scores.",
    version="1.0"
)


class StudentInput(BaseModel):
    gender: Literal["male", "female"]
    race_ethnicity: Literal["group A", "group B", "group C", "group D", "group E"]
    parental_level_of_education: Literal[
        "some high school",
        "high school",
        "some college",
        "associate's degree",
        "bachelor's degree",
        "master's degree"
    ]
    lunch: Literal["standard", "free/reduced"]
    test_preparation_course: Literal["none", "completed"]


@app.get("/")
def home():
    return {
        "message": "Student Performance Prediction API is running successfully."
    }


@app.get("/options")
def options():
    return {
        "gender": ["male", "female"],
        "race_ethnicity": ["group A", "group B", "group C", "group D", "group E"],
        "parental_level_of_education": [
            "some high school",
            "high school",
            "some college",
            "associate's degree",
            "bachelor's degree",
            "master's degree"
        ],
        "lunch": ["standard", "free/reduced"],
        "test_preparation_course": ["none", "completed"]
    }


@app.post("/predict")
def predict_score(data: StudentInput):
    input_data = pd.DataFrame([{
        "gender": data.gender,
        "race/ethnicity": data.race_ethnicity,
        "parental level of education": data.parental_level_of_education,
        "lunch": data.lunch,
        "test preparation course": data.test_preparation_course
    }])

    prediction = model.predict(input_data)[0]

    return {
        "predicted_math_score": round(float(prediction[0]), 2),
        "predicted_reading_score": round(float(prediction[1]), 2),
        "predicted_writing_score": round(float(prediction[2]), 2)
    }