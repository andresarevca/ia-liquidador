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
# Pre-Informe
# ---------------------------------------------------------------------------

class MetadatosPreinforme(BaseModel):
    """Datos del proceso de gestión que el backend Django provee al servicio IA."""
    nro_siniestro: str = Field("", description="Número de siniestro de la aseguradora")
    solicitante: str = Field("", description="Nombre de la aseguradora solicitante")
    asegurado: str = Field("", description="Nombre del asegurado (empresa o persona)")
    fecha_denuncia: str = Field("", description="Fecha de denuncia del siniestro (ISO 8601)")
    fecha_designacion: str = Field("", description="Fecha de designación del liquidador")
    ultima_documentacion: str = Field("", description="Fecha de la última documentación recibida")
    gastos_medicos_usd: float | None = Field(None, description="Monto estimado gastos médicos en USD")
    exposicion_maxima_rc_usd: float | None = Field(None, description="Exposición máxima RC en USD")
    documentos: list[str] = Field(default_factory=list, description="Nombres de archivos entregados")
    actuaciones: list[dict[str, Any]] = Field(
        default_factory=list,
        description='Lista de actuaciones: [{"fecha": "DD/MM/YYYY", "descripcion": "..."}]',
    )


class GenerarPreinformeRequest(BaseModel):
    caso_id: str = Field(..., description="Identificador único del caso")
    paso_b: dict[str, Any] = Field(..., description="Resultado del Paso B (extracción de variables)")
    paso_c: dict[str, Any] | None = Field(None, description="Resultado del Paso C (dictamen sugerido)")
    metadatos: MetadatosPreinforme = Field(default_factory=MetadatosPreinforme)


class PreinformeData(BaseModel):
    """Datos estructurados para generar el pre-informe de liquidación."""
    # Identificadores
    caso_id: str
    nro_siniestro: str
    generado_en: str

    # Cabecera
    solicitante: str
    asegurado: str
    nro_poliza: str
    seccion: str
    periodo_desde: str
    periodo_hasta: str

    # Siniestro
    detalle_siniestro: str
    lugar_siniestro: str
    fecha_siniestro: str
    condicion_climatica: str
    fecha_denuncia: str
    fecha_designacion: str
    ultima_documentacion: str

    # Vehículos y conductores
    vehiculo_asegurado: dict[str, Any]
    conductor_asegurado: dict[str, Any]
    vehiculo_tercero: dict[str, Any] | None
    conductor_tercero: dict[str, Any] | None

    # Indemnización
    corresponde_indemnizar_asegurado: bool
    estimacion_danios_materiales: str | None
    estimacion_gastos_medicos: str | None
    razon_cobertura_asegurado: str
    corresponde_indemnizar_tercero: bool
    exposicion_maxima_rc: str | None
    razon_no_cobertura_tercero: str

    # Documentación y actuaciones
    documentacion_complementaria: list[dict[str, Any]]
    actuaciones: list[dict[str, Any]]

    # Coberturas de la póliza
    coberturas_afectadas: list[dict[str, Any]]
    poliza_cobertura_tipo: str
    poliza_suma_asegurada: str | None
    franquicia: str
    exposicion_maxima_total: str | None

    # Narrativo
    dinamica_siniestro: str
    conclusion_pericial: str
    responsabilidad_conclusion: str
    cobertura_aplica: bool | str
    franquicia_aplica: bool
    monto_sugerido_liquidar: str | None

    # Legal
    infracciones_detectadas: list[dict[str, Any]]
    normas_legales_afectadas: list[str]

    # Resultado
    responsabilidad_sugerida: str
    porcentaje_asegurado: int | None
    porcentaje_tercero: int | None
    alertas_liquidador: list[str]
    confianza_dictamen: float
    calidad_extraccion: dict[str, Any]
    conflictos: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    estado: str = "ok"
    gemini_configurado: bool
    rag_disponible: bool
    version: str = "1.0.0"
