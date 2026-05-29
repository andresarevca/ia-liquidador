"""
app/routers/preinforme.py
=========================
Endpoint para generación de datos estructurados del pre-informe de liquidación.

POST /api/preinforme/generar
    Recibe paso_b + paso_c + metadatos del caso.
    Retorna PreinformeData con todos los campos listos para el template.
"""

from fastapi import APIRouter, Depends

from app.dependencies import verificar_api_key
from app.schemas import GenerarPreinformeRequest, PreinformeData

router = APIRouter(prefix="/api/preinforme", tags=["Pre-Informe"])


@router.post(
    "/generar",
    response_model=PreinformeData,
    summary="Generar datos estructurados para el pre-informe",
    dependencies=[Depends(verificar_api_key)],
)
async def generar_preinforme(body: GenerarPreinformeRequest):
    """
    Transforma los resultados del pipeline (Paso B + Paso C) más metadatos del caso
    en el conjunto de campos estructurados necesarios para renderizar el pre-informe.

    **Llamado por el backend Django** después de que el pipeline IA complete,
    con los datos de `ResultadoIA` (paso_b, paso_c) y metadatos de `Carpeta`.
    """
    from services.preinforme import generar_datos_preinforme

    datos = generar_datos_preinforme(
        caso_id=body.caso_id,
        paso_b=body.paso_b,
        paso_c=body.paso_c,
        metadatos=body.metadatos.model_dump() if body.metadatos else {},
    )
    return PreinformeData(**datos)
