import logging
import math
import os
from contextlib import asynccontextmanager

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger("uvicorn.error")

# Apuntamos directamente a la carpeta local donde se guardó el modelo
MODEL_URI = os.getenv("MODEL_URI", "model")

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = None
    logger.info("Cargando modelo desde artefacto local: %s", MODEL_URI)
    try:
        # Cargamos directamente de la ruta local. No necesitamos set_tracking_uri.
        app.state.model = mlflow.pyfunc.load_model(MODEL_URI)
    except Exception:
        logger.exception("No se pudo cargar el modelo; revisa la ruta local de artefactos")
        raise
    logger.info("Modelo cargado correctamente")
    try:
        yield
    finally:
        app.state.model = None

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Ajusta según tus variables de entorno
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

class CarPredictionRequest(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    Fuel_Type: float
    Mileage_km: float
    Year: float
    Power_hp: float
    Engine_Size_cc: float
    Cylinders: float


class CarPredictionResponse(BaseModel):
    predicted_price: float
    currency: str
    model: str


def get_model(request: Request):
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    return model

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=CarPredictionResponse)
def predict_car_price(car: CarPredictionRequest, request: Request):
    model = get_model(request)
    try:
        input_data = pd.DataFrame([car.model_dump()])
        prediction = model.predict(input_data)
        predicted_price = float(prediction[0])
        
        if not math.isfinite(predicted_price):
            raise ValueError("El modelo produjo una predicción no finita")
            
        return CarPredictionResponse(
            predicted_price=predicted_price,
            currency="USD",
            model="local_artifact_v1"
        )
    except Exception:
        logger.exception("Error realizando predicción")
        raise HTTPException(status_code=500, detail="Error realizando predicción") from None