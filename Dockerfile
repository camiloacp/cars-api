# ==========================================
# ETAPA 1: Entrenamiento
# ==========================================
FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos la carpeta de datos y el script
COPY data/ ./data/
COPY train.py .

# Se ejecuta el entrenamiento y guarda el artefacto en /build/model/
RUN python train.py

# ==========================================
# ETAPA 2: API de Producción (Servicio)
# ==========================================
FROM python:3.11-slim 

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Traemos el modelo generado de la etapa anterior a /app/model
COPY --from=builder /build/model ./model
COPY main.py .

# CONFIGURAR LA CONFIANZA DE SKOPS EN PRODUCCIÓN
ENV MLFLOW_SKOPS_TRUSTED_TYPES=True

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]