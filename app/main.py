"""
app/main.py
===========
Punto de entrada del microservicio FastAPI para el pipeline IA de liquidación.

Arrancar con:
    uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routers import corpus, pipeline, preinforme

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicialización al arrancar el servicio."""
    settings = get_settings()

    # Crear directorio de uploads
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)

    # Configurar Gemini si hay API key
    if settings.gemini_api_key:
        from services import gemini_client
        gemini_client.configure(settings.gemini_api_key)
        logger.info("Gemini API configurada.")
    else:
        logger.warning("GEMINI_API_KEY no configurada. El pipeline no funcionará hasta que se configure.")

    logger.info("Microservicio ia-liquidador iniciado.")
    yield
    logger.info("Microservicio ia-liquidador detenido.")


app = FastAPI(
    title="IA Liquidador — Pipeline de Siniestros",
    description=(
        "Microservicio de inteligencia artificial para clasificación, extracción "
        "y dictamen sugerido de carpetas de siniestros vehiculares (Paraguay, Ley 5016/14)."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — ajustar origins en producción
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "http://localhost:8000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pipeline.router)
app.include_router(corpus.router)
app.include_router(preinforme.router)


@app.get("/", include_in_schema=False)
async def root():
    return {"servicio": "ia-liquidador", "version": "1.0.0", "docs": "/docs"}
