# app.py
import os
import joblib
import zipfile
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import pandas as pd

# -----------------------
# Unzip models if needed
# -----------------------
if not (os.path.exists("fertilizer_model.pkl") and os.path.exists("soil_quality_model.pkl")):
    if os.path.exists("models.zip"):
        with zipfile.ZipFile("models.zip", "r") as zip_ref:
            zip_ref.extractall(".")
        print("✅ Models extracted from models.zip")
    else:
        raise FileNotFoundError("❌ models.zip not found and .pkl files are missing.")

# -----------------------
# Load models
# -----------------------
fertilizer_model = joblib.load("fertilizer_model.pkl")
soil_quality_model = joblib.load("soil_quality_model.pkl")

# -----------------------
# Define input schema
# -----------------------
class SoilInput(BaseModel):
    Temparature: float
    Humidity: float
    Moisture: float
    Nitrogen: float
    Phosphorous: float
    Potassium: float
    Soil_Type: str

# -----------------------
# Init API
# -----------------------
app = FastAPI()

@app.post("/predict")
def predict(data: SoilInput):
    # Prepare input as DataFrame (to match training format)
    df = pd.DataFrame([data.dict()])

    # Fertilizer prediction
    fert_pred = fertilizer_model.predict(df)[0]

    # Soil quality prediction
    quality_pred = soil_quality_model.predict(df)[0]

    return {
        "soil_quality": float(quality_pred),
        "recommended_fertilizer": str(fert_pred)
    }

# -----------------------
# Run locally
# -----------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
