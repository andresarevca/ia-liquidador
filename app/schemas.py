"""
app/schemas.py
==============
Modelos Pydantic para request/response de la API.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class JobCreado(BaseModel):
    job_id: str
    caso_id: str
    mensaje: str = "Pipeline encolado. Consulta el estado en /api/pipeline/job/{job_id}"


class JobEstado(BaseModel):
    job_id: str
    caso_id: str
    estado: str  # PENDIENTE | PROCESANDO | COMPLETADO | ERROR
    creado_en: datetime
    completado_en: datetime | None = None
    resultado: dict[str, Any] | None = None
    error: str | None = None


class ResultadoPipeline(BaseModel):
    caso_id: str
    paso_a: list[dict[str, Any]]
    paso_b: dict[str, Any]
    paso_c: dict[str, Any]
    tiempo_segundos: float
    error: str | None = None


# ---------------------------------------------------------------------------
# Corpus Legal
# ---------------------------------------------------------------------------

class IndexarTextoRequest(BaseModel):
    titulo: str = Field(..., description="Título del documento legal (ej: 'Ley 5016/14')")
    texto: str = Field(..., description="Contenido textual del documento")
    municipio: str = Field("Nacional", description="Municipio al que aplica la norma")
    tipo: str = Field("LEY", description="Tipo de documento: LEY | ORDENANZA | DECRETO")


class IndexarTextoResponse(BaseModel):
    titulo: str
    chunks_indexados: int
    municipio: str


class BusquedaLegalResult(BaseModel):
    fuente: str
    municipio: str
    encabezado: str
    texto: str
    similitud: float


class BusquedaLegalResponse(BaseModel):
    consulta: str
    resultados: list[BusquedaLegalResult]
    total: int


class FuentesResponse(BaseModel):
    fuentes: list[str]
    total_chunks: int
    total_casos_historicos: int


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    estado: str = "ok"
    gemini_configurado: bool
    rag_disponible: bool
    version: str = "1.0.0"
