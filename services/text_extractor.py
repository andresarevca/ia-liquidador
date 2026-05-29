"""
services/text_extractor.py
==========================
Extracción de texto de documentos por formato.
Soporta: PDF (pdfplumber/PyPDF2), DOCX, ODT, texto plano.
Las imágenes devuelven una marca para que Gemini las procese vía OCR nativo.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

EXTENSIONES_SOPORTADAS = {
    ".pdf":  "PDF",
    ".jpg":  "IMAGEN",
    ".jpeg": "IMAGEN",
    ".png":  "IMAGEN",
    ".gif":  "IMAGEN",
    ".webp": "IMAGEN",
    ".txt":  "TEXTO_PLANO",
    ".docx": "WORD_DOC",
    ".odt":  "WORD_DOC",
}

# Extensiones que pueden subirse a Gemini File API para OCR nativo
EXTENSIONES_OCR = {".pdf", ".jpg", ".jpeg", ".png", ".gif", ".webp"}


def get_formato(nombre_archivo: str) -> str:
    ext = Path(nombre_archivo).suffix.lower()
    return EXTENSIONES_SOPORTADAS.get(ext, "OTRO")


def extraer_texto(ruta: str, nombre_archivo: str) -> str:
    """
    Extrae el contenido textual de un archivo.
    Para imágenes y PDFs escaneados sin texto, devuelve un marcador
    indicando que Gemini debe procesarlo con OCR nativo.
    """
    formato = get_formato(nombre_archivo)

    if formato == "TEXTO_PLANO":
        try:
            return Path(ruta).read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.warning(f"Error leyendo texto plano {nombre_archivo}: {e}")
            return ""

    if formato == "PDF":
        texto = _extraer_pdf(ruta, nombre_archivo)
        if texto:
            return texto
        return f"[PDF sin texto extraíble: {nombre_archivo} — se usará OCR de Gemini]"

    if formato == "IMAGEN":
        return f"[Imagen: {nombre_archivo} — se procesará con OCR de Gemini]"

    if formato == "WORD_DOC":
        ext = Path(nombre_archivo).suffix.lower()
        if ext == ".docx":
            return _extraer_docx(ruta, nombre_archivo)
        if ext == ".odt":
            return _extraer_odt(ruta, nombre_archivo)

    return f"[Formato no soportado para extracción de texto: {nombre_archivo}]"


def _extraer_pdf(ruta: str, nombre: str) -> str:
    """Intenta pdfplumber primero, luego PyPDF2 como fallback."""
    try:
        import pdfplumber
        with pdfplumber.open(ruta) as pdf:
            texto = "\n".join(p.extract_text() or "" for p in pdf.pages).strip()
        if texto:
            return texto
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"pdfplumber falló en {nombre}: {e}")

    try:
        import PyPDF2
        with open(ruta, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            texto = "\n".join((p.extract_text() or "") for p in reader.pages).strip()
        if texto:
            return texto
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"PyPDF2 falló en {nombre}: {e}")

    return ""


def _extraer_docx(ruta: str, nombre: str) -> str:
    try:
        import docx
        doc = docx.Document(ruta)
        return "\n".join(p.text for p in doc.paragraphs).strip()
    except ImportError:
        return f"[DOCX sin extracción: instalar python-docx]"
    except Exception as e:
        logger.warning(f"python-docx falló en {nombre}: {e}")
        return ""


def _extraer_odt(ruta: str, nombre: str) -> str:
    try:
        from odf.opendocument import load as odf_load
        from odf.text import P
        doc = odf_load(ruta)
        return "\n".join(str(el) for el in doc.getElementsByType(P)).strip()
    except ImportError:
        return f"[ODT sin extracción: instalar odfpy]"
    except Exception as e:
        logger.warning(f"odfpy falló en {nombre}: {e}")
        return ""
