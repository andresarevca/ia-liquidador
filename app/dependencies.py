"""
app/dependencies.py
===================
Dependencias de FastAPI: autenticación y configuración.
"""

from fastapi import Header, HTTPException, status

from app.config import get_settings


async def verificar_api_key(x_api_key: str = Header(default="")) -> None:
    """
    Verifica el header X-Api-Key si API_KEY está configurada en .env.
    Si API_KEY no está configurada, no se aplica autenticación (útil en desarrollo).
    """
    settings = get_settings()
    if not settings.api_key:
        return  # Sin restricción en desarrollo
    if x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key inválida o ausente. Incluye el header X-Api-Key.",
        )
