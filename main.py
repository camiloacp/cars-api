import logging
import math
import os
from contextlib import asynccontextmanager

import mlflow
import mlflow.pyfunc
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict

# Las variables inyectadas por Docker tienen prioridad sobre el archivo local.
load_dotenv(override=False)
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "autos_mejor_modelo")
PRODUCTION_ALIAS = os.getenv("PRODUCTION_ALIAS", "production")
MODEL_URI = f"models:/{REGISTERED_MODEL_NAME}@{PRODUCTION_ALIAS}"
logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = None
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    logger.info("Cargando modelo: %s", MODEL_URI)
    try:
        app.state.model = mlflow.pyfunc.load_model(MODEL_URI)
    except Exception:
        logger.exception("No se pudo cargar el modelo; revisa MLflow, el alias y los artefactos")
        raise
    logger.info("Modelo cargado correctamente")
    try:
        yield
    finally:
        app.state.model = None


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.getenv("CORS_ALLOWED_ORIGINS", "").split(",")
        if origin.strip()
    ],
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


@app.get("/")
def index():
    return {"message": "Cars Api"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready(request: Request):
    get_model(request)
    return {"status": "ready", "model": f"{REGISTERED_MODEL_NAME}@{PRODUCTION_ALIAS}"}


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
            model=f"{REGISTERED_MODEL_NAME}@{PRODUCTION_ALIAS}",
        )
    except Exception:
        logger.exception("Error realizando predicción")
        raise HTTPException(status_code=500, detail="Error realizando predicción") from None
