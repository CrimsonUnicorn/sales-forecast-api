from fastapi import FastAPI, UploadFile, File
import pandas as pd
from model import train_models, predict_sales

app = FastAPI()

models = None

@app.get("/")
def home():
    return {"message": "Sales Forecast API running"}

@app.post("/upload-data")
async def upload_data(file: UploadFile = File(...)):
    global models

    df = pd.read_csv(file.file)

    models = train_models(df)

    return {"message": "Models trained successfully"}

@app.get("/models")
def list_models():

    global models

    if models is None:
        return {"error": "Train models first"}

    return {
        "available_models": list(models.keys())
    }

@app.get("/forecast")
def forecast(day: int, model: str):

    global models

    if models is None:
        return {"error": "Upload data first"}

    prediction = predict_sales(models, model, day)

    if prediction is None:
        return {"error": "Invalid model name"}

    return {
        "model": model,
        "day": day,
        "predicted_sales": prediction
    }