"""
app/routers/pipeline.py
=======================
Endpoints del pipeline de liquidación de siniestros.

POST /api/pipeline/ejecutar
    Recibe archivos del caso, ejecuta A→B→C de forma síncrona.
    Usado por el backend Django desde una tarea Celery.

POST /api/pipeline/ejecutar-async
    Encola el pipeline y retorna job_id de inmediato.

GET  /api/pipeline/job/{job_id}
    Consulta el estado de un job asíncrono.

GET  /api/pipeline/health
    Verificación de disponibilidad del servicio.
"""

import os
import shutil
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.config import get_settings
from app.dependencies import verificar_api_key
from app.schemas import HealthResponse, JobCreado, JobEstado, ResultadoPipeline

router = APIRouter(prefix="/api/pipeline", tags=["Pipeline IA"])

# ---------------------------------------------------------------------------
# Almacén en memoria de jobs asíncronos
# ---------------------------------------------------------------------------
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()

_EXTENSIONES_PERMITIDAS = {".pdf", ".jpg", ".jpeg", ".png", ".gif", ".webp", ".txt", ".docx", ".odt"}


def _guardar_job(job_id: str, caso_id: str) -> None:
    with _jobs_lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "caso_id": caso_id,
            "estado": "PENDIENTE",
            "creado_en": datetime.utcnow(),
            "completado_en": None,
            "resultado": None,
            "error": None,
        }


def _actualizar_job(job_id: str, **kwargs) -> None:
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(kwargs)


def _obtener_job(job_id: str) -> dict | None:
    with _jobs_lock:
        return _jobs.get(job_id)


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def _validar_archivos(archivos: list[UploadFile], max_mb: int) -> None:
    max_bytes = max_mb * 1024 * 1024
    for archivo in archivos:
        ext = Path(archivo.filename or "").suffix.lower()
        if ext not in _EXTENSIONES_PERMITIDAS:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Extensión no permitida: '{archivo.filename}'. Permitidas: {', '.join(_EXTENSIONES_PERMITIDAS)}",
            )
        # Nota: el tamaño real se verifica al guardar


def _guardar_archivos_temp(archivos: list[UploadFile], directorio: Path) -> list[dict]:
    """Guarda archivos en disco temporal y retorna la lista para el pipeline."""
    resultado = []
    for archivo in archivos:
        nombre = Path(archivo.filename or f"archivo_{uuid.uuid4().hex[:8]}").name
        ruta = directorio / nombre
        with open(ruta, "wb") as f:
            shutil.copyfileobj(archivo.file, f)
        resultado.append({"ruta": str(ruta), "nombre": nombre})
    return resultado


def _ejecutar_y_limpiar(job_id: str, caso_id: str, directorio: Path, usar_rag: bool) -> None:
    """Worker de thread: ejecuta el pipeline y limpia archivos temporales."""
    from services import gemini_client
    from services.pipeline import ejecutar_pipeline

    try:
        settings = get_settings()
        gemini_client.configure(settings.gemini_api_key)

        _actualizar_job(job_id, estado="PROCESANDO")

        archivos = [
            {"ruta": str(f), "nombre": f.name}
            for f in directorio.iterdir()
            if f.is_file() and f.suffix.lower() in _EXTENSIONES_PERMITIDAS
        ]

        resultado = ejecutar_pipeline(caso_id, archivos, usar_rag=usar_rag)

        if resultado.get("error") and not resultado.get("paso_a"):
            _actualizar_job(
                job_id,
                estado="ERROR",
                error=resultado["error"],
                completado_en=datetime.utcnow(),
            )
        else:
            _actualizar_job(
                job_id,
                estado="COMPLETADO",
                resultado=resultado,
                completado_en=datetime.utcnow(),
            )
    except Exception as e:
        _actualizar_job(
            job_id,
            estado="ERROR",
            error=str(e),
            completado_en=datetime.utcnow(),
        )
    finally:
        shutil.rmtree(directorio, ignore_errors=True)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse, summary="Verificación de disponibilidad")
async def health():
    settings = get_settings()
    try:
        from vector_store import CorpusLegal  # noqa: F401
        rag_ok = True
    except ImportError:
        rag_ok = False

    return HealthResponse(
        gemini_configurado=bool(settings.gemini_api_key),
        rag_disponible=rag_ok,
    )


@router.post(
    "/ejecutar",
    response_model=ResultadoPipeline,
    summary="Ejecutar pipeline completo (síncrono)",
    dependencies=[Depends(verificar_api_key)],
)
async def ejecutar_pipeline_sync(
    caso_id: Annotated[str, Form(description="Identificador único del caso")],
    archivos: Annotated[list[UploadFile], File(description="Documentos del caso (PDF, imágenes, DOCX, TXT)")],
    usar_rag: Annotated[bool, Form(description="Activar búsqueda semántica en corpus legal (requiere ChromaDB)")] = False,
):
    """
    Ejecuta el pipeline A→B→C de forma síncrona y retorna los resultados completos.
    **Recomendado para integraciones server-to-server** (ej: Celery worker de Django).
    """
    from services import gemini_client
    from services.pipeline import ejecutar_pipeline

    settings = get_settings()
    if not settings.gemini_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GEMINI_API_KEY no configurada en el servidor.",
        )

    _validar_archivos(archivos, settings.max_file_size_mb)

    directorio = Path(settings.upload_dir) / f"{caso_id}_{uuid.uuid4().hex[:8]}"
    directorio.mkdir(parents=True, exist_ok=True)

    try:
        gemini_client.configure(settings.gemini_api_key)
        lista_archivos = _guardar_archivos_temp(archivos, directorio)
        resultado = ejecutar_pipeline(caso_id, lista_archivos, usar_rag=usar_rag)
    finally:
        shutil.rmtree(directorio, ignore_errors=True)

    if resultado.get("error") and not resultado.get("paso_a"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=resultado["error"],
        )

    return ResultadoPipeline(
        caso_id=caso_id,
        paso_a=resultado.get("paso_a", []),
        paso_b=resultado.get("paso_b", {}),
        paso_c=resultado.get("paso_c", {}),
        tiempo_segundos=resultado.get("tiempo_segundos", 0.0),
        error=resultado.get("error"),
    )


@router.post(
    "/ejecutar-async",
    response_model=JobCreado,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ejecutar pipeline en segundo plano (asíncrono)",
    dependencies=[Depends(verificar_api_key)],
)
async def ejecutar_pipeline_async(
    caso_id: Annotated[str, Form()],
    archivos: Annotated[list[UploadFile], File()],
    usar_rag: Annotated[bool, Form()] = False,
):
    """
    Encola el pipeline y retorna un `job_id` de inmediato.
    Consulta el estado con `GET /api/pipeline/job/{job_id}`.
    """
    settings = get_settings()
    if not settings.gemini_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GEMINI_API_KEY no configurada.",
        )

    _validar_archivos(archivos, settings.max_file_size_mb)

    job_id = uuid.uuid4().hex
    directorio = Path(settings.upload_dir) / f"{caso_id}_{job_id[:8]}"
    directorio.mkdir(parents=True, exist_ok=True)

    _guardar_archivos_temp(archivos, directorio)
    _guardar_job(job_id, caso_id)

    hilo = threading.Thread(
        target=_ejecutar_y_limpiar,
        args=(job_id, caso_id, directorio, usar_rag),
        daemon=True,
    )
    hilo.start()

    return JobCreado(job_id=job_id, caso_id=caso_id)


@router.get(
    "/job/{job_id}",
    response_model=JobEstado,
    summary="Consultar estado de un job asíncrono",
    dependencies=[Depends(verificar_api_key)],
)
async def estado_job(job_id: str):
    """Retorna el estado y resultado (si completó) de un job asíncrono."""
    job = _obtener_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' no encontrado.",
        )
    return JobEstado(**job)
