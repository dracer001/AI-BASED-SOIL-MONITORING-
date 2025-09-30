# app.py
import os
import joblib
import zipfile
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import pandas as pd

# Init API
app = FastAPI()

# ✅ Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # or restrict to ["https://hoppscotch.io"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    Soil_Type: str  # <-- keep underscore here


# -----------------------
# Init API
# -----------------------
app = FastAPI()

@app.post("/predict")
def predict(data: SoilInput):
    df = pd.DataFrame([data.dict()])
    # Rename columns to match training data
    df = df.rename(columns={"Soil_Type": "Soil Type"})

    fert_pred = fertilizer_model.predict(df)[0]
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
