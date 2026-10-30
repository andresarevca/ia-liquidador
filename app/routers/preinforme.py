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


def _nulos_a_vacio(datos: dict) -> dict:
    """Gemini puede devolver `null` en campos que el esquema exige como texto.

    `dict.get(clave, "")` solo usa el valor por defecto si la clave no existe,
    no si vale None. Se normalizan aqui los campos `str` obligatorios.
    """
    for nombre, campo in PreinformeData.model_fields.items():
        if campo.annotation is str and datos.get(nombre) is None:
            datos[nombre] = ""
    return datos


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
    return PreinformeData(**_nulos_a_vacio(datos))
