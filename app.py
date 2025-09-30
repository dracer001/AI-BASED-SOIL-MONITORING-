# app.py
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# Load models
fertilizer_model = joblib.load("fertilizer_model.pkl")
soil_quality_model = joblib.load("soil_quality_model.pkl")

# Define input schema
class SoilInput(BaseModel):
    Temparature: float
    Humidity: float
    Moisture: float
    Nitrogen: float
    Phosphorous: float
    Potassium: float
    Soil_Type: str

# Init API
app = FastAPI()

@app.post("/predict")
def predict(data: SoilInput):
    # Prepare input as dataframe (to match training format)
    import pandas as pd
    df = pd.DataFrame([data.dict()])

    # Fertilizer prediction
    fert_pred = fertilizer_model.predict(df)[0]

    # Soil quality prediction
    quality_pred = soil_quality_model.predict(df)[0]

    return {
        "soil_quality": float(quality_pred),
        "recommended_fertilizer": str(fert_pred)
    }

# For local testing
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
