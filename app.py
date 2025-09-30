# app.py
import os
import zipfile
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# === Unzip models if not already extracted ===
if not os.path.exists("models"):
    with zipfile.ZipFile("models.zip", "r") as zip_ref:
        zip_ref.extractall("models")

# === Load models from extracted folder ===
fertilizer_model = joblib.load("models/fertilizer_model.pkl")
soil_quality_model = joblib.load("models/soil_quality_model.pkl")

# === Define input schema ===
class SoilInput(BaseModel):
    Temparature: float
    Humidity: float
    Moisture: float
    Nitrogen: float
    Phosphorous: float
    Potassium: float
    Soil_Type: str   # make sure your model was trained with string/categorical features handled

# === Initialize FastAPI ===
app = FastAPI()

@app.post("/predict")
def predict(data: SoilInput):
    # Convert input to dataframe
    df = pd.DataFrame([data.dict()])

    # Run predictions
    fert_pred = fertilizer_model.predict(df)[0]
    quality_pred = soil_quality_model.predict(df)[0]

    return {
        "soil_quality": float(quality_pred),
        "recommended_fertilizer": str(fert_pred)
    }

# === Local testing only ===
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
