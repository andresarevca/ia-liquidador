"""
app/routers/corpus.py
=====================
Endpoints para gestión del corpus legal (vector store ChromaDB + RAG).

POST /api/corpus/indexar-texto    — Indexa un texto legal (ley u ordenanza)
POST /api/corpus/indexar-pdf      — Sube e indexa un PDF de ordenanza
GET  /api/corpus/buscar           — Búsqueda semántica en el corpus
GET  /api/corpus/fuentes          — Lista documentos indexados
POST /api/corpus/indexar-historicos — Reconstruye índice desde resultados previos
"""

import shutil
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status

from app.config import get_settings
from app.dependencies import verificar_api_key
from app.schemas import (
    BusquedaLegalResponse,
    BusquedaLegalResult,
    FuentesResponse,
    IndexarTextoRequest,
    IndexarTextoResponse,
)

router = APIRouter(prefix="/api/corpus", tags=["Corpus Legal (RAG)"])


def _get_corpus():
    """Instancia CorpusLegal con la ruta configurada."""
    try:
        from vector_store import CorpusLegal
        settings = get_settings()
        return CorpusLegal(db_path=settings.vector_db_path)
    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ChromaDB no disponible. Instalar con: pip install chromadb",
        )


def _get_casos():
    """Instancia CasosHistoricos con la ruta configurada."""
    try:
        from vector_store import CasosHistoricos
        settings = get_settings()
        return CasosHistoricos(db_path=settings.vector_db_path)
    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ChromaDB no disponible. Instalar con: pip install chromadb",
        )


def _configurar_gemini():
    settings = get_settings()
    if not settings.gemini_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GEMINI_API_KEY no configurada.",
        )
    from services import gemini_client
    gemini_client.configure(settings.gemini_api_key)


@router.post(
    "/indexar-texto",
    response_model=IndexarTextoResponse,
    summary="Indexar texto legal (ley u ordenanza)",
    dependencies=[Depends(verificar_api_key)],
)
async def indexar_texto(body: IndexarTextoRequest):
    """
    Indexa un documento legal en el corpus semántico.
    El texto se chunkeará automáticamente por artículos.
    Es idempotente: no duplica chunks ya existentes.
    """
    _configurar_gemini()
    corpus = _get_corpus()
    n = corpus.indexar_texto(body.titulo, body.texto, municipio=body.municipio, tipo=body.tipo)
    return IndexarTextoResponse(titulo=body.titulo, chunks_indexados=n, municipio=body.municipio)


@router.post(
    "/indexar-pdf",
    response_model=IndexarTextoResponse,
    summary="Subir e indexar un PDF de ordenanza municipal",
    dependencies=[Depends(verificar_api_key)],
)
async def indexar_pdf(
    archivo: Annotated[UploadFile, File(description="PDF de la ordenanza municipal")],
    municipio: Annotated[str, Query(description="Municipio al que aplica la ordenanza")] = "Nacional",
):
    """
    Sube un PDF, extrae su texto y lo indexa en el corpus legal.
    Útil para cargar ordenanzas de distintos municipios de Paraguay.
    """
    _configurar_gemini()

    if not archivo.filename or not archivo.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Solo se aceptan archivos PDF.",
        )

    settings = get_settings()
    tmp_dir = Path(settings.upload_dir) / uuid.uuid4().hex[:8]
    tmp_dir.mkdir(parents=True, exist_ok=True)
    ruta = tmp_dir / archivo.filename

    try:
        with open(ruta, "wb") as f:
            shutil.copyfileobj(archivo.file, f)

        corpus = _get_corpus()
        n = corpus.indexar_pdf(ruta, municipio=municipio)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return IndexarTextoResponse(titulo=ruta.stem, chunks_indexados=n, municipio=municipio)


@router.get(
    "/buscar",
    response_model=BusquedaLegalResponse,
    summary="Búsqueda semántica en el corpus legal",
    dependencies=[Depends(verificar_api_key)],
)
async def buscar(
    q: Annotated[str, Query(description="Texto de la consulta legal", min_length=3)],
    municipio: Annotated[str | None, Query(description="Filtrar por municipio (opcional)")] = None,
    n: Annotated[int, Query(description="Número máximo de resultados", ge=1, le=20)] = 6,
):
    """Realiza una búsqueda semántica en el corpus de textos legales indexados."""
    _configurar_gemini()
    corpus = _get_corpus()
    resultados = corpus.buscar_relevantes(q, n=n, municipio=municipio)

    return BusquedaLegalResponse(
        consulta=q,
        resultados=[BusquedaLegalResult(**r) for r in resultados],
        total=len(resultados),
    )


@router.get(
    "/fuentes",
    response_model=FuentesResponse,
    summary="Listar documentos legales indexados",
    dependencies=[Depends(verificar_api_key)],
)
async def listar_fuentes():
    """Retorna la lista de documentos legales indexados y estadísticas del corpus."""
    corpus = _get_corpus()
    casos = _get_casos()
    return FuentesResponse(
        fuentes=corpus.listar_fuentes(),
        total_chunks=corpus.contar(),
        total_casos_historicos=casos.contar(),
    )


@router.post(
    "/indexar-historicos",
    summary="Reconstruir índice de casos históricos",
    dependencies=[Depends(verificar_api_key)],
)
async def indexar_historicos():
    """
    Reconstruye el índice de casos históricos desde la carpeta `resultados/`.
    Útil para inicializar el vector store con casos ya procesados.
    """
    _configurar_gemini()
    casos = _get_casos()
    n = casos.cargar_desde_resultados()
    return {"casos_indexados": n, "total_en_store": casos.contar()}
