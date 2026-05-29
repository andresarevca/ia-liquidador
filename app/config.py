"""
app/config.py
=============
Configuración del microservicio via variables de entorno.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    gemini_api_key: str = ""
    # API key que el backend Django debe enviar en el header X-API-Key
    api_key: str = ""
    # Ruta del vector store persistente (ChromaDB)
    vector_db_path: str = "/app/vector_db"
    # Límite de tamaño por archivo en MB
    max_file_size_mb: int = 20
    # Directorio temporal para archivos subidos
    upload_dir: str = "/tmp/ia-liquidador"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
