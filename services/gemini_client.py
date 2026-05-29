"""
services/gemini_client.py
=========================
Wrapper sobre google-generativeai con reintentos, parseo de JSON seguro
y subida de archivos a la File API (OCR nativo para PDFs e imágenes).
"""

import json
import logging
import time
from pathlib import Path

import google.generativeai as genai

logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-2.5-flash"
MAX_RETRIES = 3
RETRY_DELAY = 5

MIME_TYPES = {
    ".pdf":  "application/pdf",
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".gif":  "image/gif",
    ".webp": "image/webp",
}


def configure(api_key: str) -> None:
    """Configura la API key de Gemini."""
    genai.configure(api_key=api_key)


def llamar_gemini(
    prompt_sistema: str,
    prompt_usuario: str,
    partes: list | None = None,
    modelo: str = GEMINI_MODEL,
) -> str | None:
    """
    Llama a Gemini con reintentos automáticos.
    Si se proveen `partes` (archivos de la File API), se construye
    contenido multimodal: [*partes, prompt_usuario].
    """
    model = genai.GenerativeModel(
        model_name=modelo,
        system_instruction=prompt_sistema,
    )
    contenido = (list(partes) + [prompt_usuario]) if partes else prompt_usuario

    for intento in range(1, MAX_RETRIES + 1):
        try:
            respuesta = model.generate_content(contenido)
            return respuesta.text
        except Exception as e:
            if intento < MAX_RETRIES:
                logger.warning(f"Gemini intento {intento}/{MAX_RETRIES} falló ({type(e).__name__}). Reintentando en {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                logger.error(f"Gemini falló tras {MAX_RETRIES} intentos: {e}")
                return None


def subir_archivo(ruta: str, nombre: str):
    """
    Sube un archivo a la Gemini File API para OCR nativo.
    Retorna el objeto File o None si el formato no es soportado.
    """
    ext = Path(nombre).suffix.lower()
    mime = MIME_TYPES.get(ext)
    if not mime:
        return None
    try:
        logger.info(f"Subiendo a Gemini File API: {nombre} ({mime})")
        return genai.upload_file(path=ruta, mime_type=mime, display_name=nombre)
    except Exception as e:
        logger.warning(f"No se pudo subir {nombre} a File API: {e}")
        return None


def parsear_json(texto: str) -> dict | None:
    """Parsea JSON de la respuesta de Gemini, eliminando bloques Markdown si existen."""
    texto = texto.strip()
    if texto.startswith("```"):
        lineas = texto.split("\n")
        texto = "\n".join(lineas[1:])
        if texto.strip().endswith("```"):
            texto = texto.strip()[:-3]
    try:
        return json.loads(texto)
    except json.JSONDecodeError as e:
        logger.error(f"JSON inválido de Gemini: {e}. Respuesta (300 chars): {texto[:300]}")
        return None
