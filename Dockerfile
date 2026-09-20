# Imagen ligera de Python (las deps tienen wheels precompilados para 3.11)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Instalar dependencias primero para aprovechar la caché de capas
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código de la aplicación
COPY main.py .

# No correr como root
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

# Cada worker carga su propia copia del modelo durante el arranque.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
