import os

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


app = FastAPI(title="Iris Classifier API")


def load_model():
    model_dir = os.getenv("MODEL_DIR", "models/iris_classifier")
    return mlflow.sklearn.load_model(model_dir)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(features: IrisFeatures):
    model = load_model()
    data = pd.DataFrame(
        [[features.sepal_length, features.sepal_width, features.petal_length, features.petal_width]],
        columns=["sepal_length", "sepal_width", "petal_length", "petal_width"],
    )
    prediction = model.predict(data)[0]
    return {"species": prediction}
