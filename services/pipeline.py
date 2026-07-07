"""
services/pipeline.py
====================
Lógica central del pipeline de liquidación de siniestros.
Extraída y refactorizada de validar_pipeline_ia.py para su uso como servicio.

Pasos:
  A — Clasificación de documentos
  B — Extracción de variables críticas
  C — Análisis normativo y dictamen sugerido (con RAG opcional)
"""

import logging
from pathlib import Path

from services.gemini_client import GEMINI_MODEL, llamar_gemini, parsear_json, subir_archivo
from services.text_extractor import EXTENSIONES_OCR, extraer_texto, get_formato

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Marco legal integrado (base — puede enriquecerse con RAG)
# ---------------------------------------------------------------------------

LEY_5016_FRAGMENTOS = """
LEY N° 5016/14 — DE TRÁNSITO Y SEGURIDAD VIAL — REPÚBLICA DEL PARAGUAY

Art. 89 — CONDICIONES CLIMÁTICAS:
En condiciones de lluvia, el conductor deberá reducir la velocidad y aumentar
la distancia de seguridad respecto al vehículo precedente.

Art. 139 — SEÑALES DE CONTROL DE TRÁNSITO:
Los conductores están obligados a obedecer las señales de tránsito. La señal
de PARE obliga a detener completamente el vehículo antes de la línea de pare
o de la intersección y a ceder el paso a los vehículos que circulen por la vía
que se va a cruzar.

Art. 158 — RESPONSABILIDAD POR ACCIDENTES:
El conductor que infrinja las normas de tránsito y como consecuencia de dicha
infracción cause un accidente, será responsable de los daños y perjuicios
ocasionados, conforme a las disposiciones del Código Civil.

Art. 201 — LICENCIA DE CONDUCIR OBLIGATORIA:
Es obligatorio para todo conductor portar licencia de conducir habilitante
para la categoría del vehículo que conduce. La conducción sin licencia
constituye infracción muy grave.
""".strip()

ORDENANZA_SAN_LORENZO = """
MUNICIPALIDAD DE SAN LORENZO
ORDENANZA N° 45/2019 — TRÁNSITO URBANO

Art. 23 — PRIORIDAD DE PASO EN INTERSECCIONES:
En toda intersección señalizada con señal de PARE (STOP), el conductor que
enfrente dicha señal deberá detener completamente su vehículo y ceder el paso
a todo vehículo que circule por la vía preferencial, bajo pena de multa de
2 (dos) jornales mínimos y responsabilidad civil por los daños ocasionados.

Art. 31 — VELOCIDAD MÁXIMA EN ZONA URBANA:
La velocidad máxima en calles y avenidas urbanas del municipio de San Lorenzo
es de 40 km/h, salvo señalización específica que indique valor diferente.

Art. 45 — DOCUMENTACIÓN OBLIGATORIA:
Todo conductor debe portar: licencia de conducir vigente, cédula verde del
vehículo y comprobante de seguro vigente. La no presentación de licencia
constituye infracción grave sancionada con retención del vehículo.
""".strip()


# ---------------------------------------------------------------------------
# PASO A — Clasificación de documentos
# ---------------------------------------------------------------------------

_SYSTEM_A = """
Eres un clasificador especializado en documentos de siniestros vehiculares del Paraguay.
Tu única tarea es analizar el contenido de cada archivo adjunto y devolver
una clasificación estructurada en JSON.

REGLAS ESTRICTAS:
1. Devuelve ÚNICAMENTE el objeto JSON. Sin texto adicional, sin Markdown, sin explicaciones.
2. Si el documento es ilegible o de baja calidad, marca legible: false.
3. No extraigas datos del contenido en este paso. Solo clasifica.
4. Ante la duda entre dos tipos, usa OTRO y describe en nota_clasificacion.
""".strip()


def ejecutar_paso_a(documentos: list[dict]) -> list[dict]:
    """
    Clasifica cada documento.

    documentos: [{"nombre_archivo": str, "formato": str, "contenido": str, "gemini_file": obj|None}]
    Retorna lista con clasificaciones y contenido_original preservado.
    """
    resultados = []
    for doc in documentos:
        gemini_file = doc.get("gemini_file")
        if gemini_file:
            bloque_contenido = "El documento ha sido adjuntado directamente para análisis con OCR nativo."
        else:
            bloque_contenido = f"CONTENIDO DEL DOCUMENTO:\n{doc.get('contenido', '')[:6000]}"

        prompt = f"""
Analiza el siguiente documento del caso de siniestro vehicular.

CONTEXTO:
- Nombre del archivo: {doc['nombre_archivo']}
- Formato: {doc['formato']}
{bloque_contenido}

Clasifica este documento y devuelve EXCLUSIVAMENTE este JSON:

{{
  "tipo_doc": "<POLIZA | DENUNCIA_POLICIAL | DENUNCIA_ADMINISTRATIVA | PERICIA_TECNICA | PRESUPUESTO_TALLER | FOTO_EVENTO | FOTO_DANIOS | CEDULA_VERDE | LICENCIA_CONDUCIR | INFORME_MEDICO | OTRO>",
  "legible": <true | false>,
  "confianza": <0.0 a 1.0>,
  "nota_clasificacion": "<observación breve o null>",
  "pertenece_al_caso": <true | false>,
  "razon_pertenencia": "<una frase explicando por qué>"
}}""".strip()

        partes = [gemini_file] if gemini_file else None
        respuesta = llamar_gemini(_SYSTEM_A, prompt, partes=partes)

        if respuesta:
            parsed = parsear_json(respuesta)
            if parsed:
                parsed["archivo"] = doc["nombre_archivo"]
                parsed["contenido_original"] = doc.get("contenido", "")
                parsed["gemini_file"] = gemini_file
                resultados.append(parsed)
                logger.info(f"Paso A — {doc['nombre_archivo']} → {parsed.get('tipo_doc')} (conf: {parsed.get('confianza')})")
                continue

        logger.error(f"Paso A — sin resultado para {doc['nombre_archivo']}")
        resultados.append({"archivo": doc["nombre_archivo"], "error": True})

    return resultados


# ---------------------------------------------------------------------------
# PASO B — Extracción de variables críticas
# ---------------------------------------------------------------------------

_SYSTEM_B = """
Eres un extractor de datos forense especializado en siniestros vehiculares del Paraguay.
Tu tarea es analizar un conjunto de documentos de un mismo caso y extraer
las variables críticas con máxima precisión.

REGLAS ESTRICTAS:
1. Devuelve ÚNICAMENTE el objeto JSON. Sin texto adicional ni Markdown.
2. Si un dato no figura en ningún documento, usa null. NUNCA inventes datos.
3. Si el mismo dato aparece con valores contradictorios, agrégalo al array "conflictos".
4. Las fechas deben estar en formato ISO 8601: YYYY-MM-DDTHH:MM:SS.
5. Los montos siempre en Guaraníes (PYG).
""".strip()

_SCHEMA_B = """
{
  "siniestro": {
    "fecha_hora": "<ISO 8601 o null>",
    "municipio": "<municipio de Paraguay o null>",
    "direccion_aproximada": "<calle, barrio o null>",
    "tipo_via": "<CALLE_URBANA | RUTA_NACIONAL | RUTA_DEPARTAMENTAL | ESTACIONAMIENTO | OTRA | null>",
    "condicion_climatica": "<DESPEJADO | LLUVIA | NIEBLA | NOCHE | DESCONOCIDO>",
    "descripcion_dinamica": "<párrafo descriptivo en español formal>"
  },
  "vehiculos": [
    {
      "rol": "<ASEGURADO | TERCERO_1>",
      "marca": null, "modelo": null, "año": null, "matricula": null, "color": null,
      "conductor_nombre": null, "conductor_ci": null,
      "licencia_numero": null, "licencia_categoria": null,
      "danios_descripcion": null
    }
  ],
  "poliza": {
    "numero": null, "aseguradora": null,
    "vigencia_desde": null, "vigencia_hasta": null,
    "cobertura_tipo": "<RESPONSABILIDAD_CIVIL | TODO_RIESGO | PARCIAL | DESCONOCIDO>",
    "suma_asegurada": null, "franquicia": null
  },
  "documentacion": {
    "poliza_presente": false, "denuncia_policial_presente": false,
    "denuncia_administrativa_presente": false, "pericia_tecnica_presente": false,
    "fotos_evento_cantidad": 0, "cedula_verde_presente": false,
    "licencia_conductor_presente": false
  },
  "monto_danios": { "estimacion_pericia": null, "moneda_original": "PYG" },
  "conflictos": [
    {
      "campo": "<nombre del campo>",
      "valor_doc_1": "<valor según documento A>",
      "valor_doc_2": "<valor según documento B>",
      "descripcion": "<explicación>"
    }
  ],
  "calidad_extraccion": {
    "score": 0.0,
    "campos_faltantes_criticos": [],
    "observaciones": null
  }
}
""".strip()


def ejecutar_paso_b(resultados_a: list[dict], caso_id: str) -> dict | None:
    """Extrae variables críticas del conjunto de documentos clasificados."""
    hay_ocr = any(r.get("gemini_file") for r in resultados_a if not r.get("error"))

    if hay_ocr:
        partes = []
        for r in resultados_a:
            if r.get("error"):
                continue
            partes.append(f"--- DOCUMENTO: {r['archivo']} | TIPO: {r.get('tipo_doc', '?')} ---")
            if r.get("gemini_file"):
                partes.append(r["gemini_file"])
            else:
                partes.append(r.get("contenido_original", ""))

        prompt = f"Analiza los documentos del caso {caso_id} adjuntos y extrae variables.\n\nDevuelve EXCLUSIVAMENTE este JSON:\n{_SCHEMA_B}"
    else:
        bloques = []
        for r in resultados_a:
            if r.get("error"):
                continue
            bloques.append(
                f"--- DOCUMENTO: {r['archivo']} | TIPO: {r.get('tipo_doc', '?')} ---\n"
                f"{r.get('contenido_original', '')[:4000]}"
            )
        partes = None
        prompt = (
            f"Analiza los siguientes documentos del caso {caso_id} y extrae variables estructuradas.\n\n"
            f"DOCUMENTOS DEL CASO:\n{''.join(bloques)}\n\n"
            f"Devuelve EXCLUSIVAMENTE este JSON:\n{_SCHEMA_B}"
        )

    respuesta = llamar_gemini(_SYSTEM_B, prompt, partes=partes)
    if respuesta:
        result = parsear_json(respuesta)
        if result:
            score = result.get("calidad_extraccion", {}).get("score", 0)
            logger.info(f"Paso B — caso {caso_id} | score: {score}")
        return result
    return None


# ---------------------------------------------------------------------------
# PASO C — Análisis normativo y dictamen sugerido
# ---------------------------------------------------------------------------

_SYSTEM_C = """
Eres un analista jurídico especializado en derecho de tránsito del Paraguay.
Tu tarea es analizar los datos de un siniestro vehicular y emitir una sugerencia
de dictamen fundamentada en la normativa vigente.

MARCO LEGAL APLICABLE:
- Ley N° 5016/14 "De Tránsito y Seguridad Vial" (nacional)
- Ordenanza municipal del municipio del siniestro (provista en contexto)
- Código Civil Paraguayo: Arts. 1833-1847 (responsabilidad extracontractual)

REGLAS ESTRICTAS:
1. Devuelve ÚNICAMENTE el objeto JSON. Sin texto adicional ni Markdown.
2. NUNCA emitas un dictamen definitivo. Tu salida es una SUGERENCIA para el liquidador.
3. Cita los artículos específicos que fundamentan cada conclusión.
4. El campo analisis_narrativo debe estar en español formal, tercera persona.
5. Si hay conflictos en los datos del Paso B, mencionarlos obligatoriamente.
""".strip()

_SCHEMA_C = """
{
  "dictamen": {
    "dictamen_posible": true,
    "datos_faltantes_para_dictamen": [],
    "responsabilidad_sugerida": "<ASEGURADO_RESPONSABLE | TERCERO_RESPONSABLE | RESPONSABILIDAD_COMPARTIDA | CASO_FORTUITO | INDETERMINADO>",
    "porcentaje_responsabilidad_asegurado": 0,
    "porcentaje_responsabilidad_tercero": 0,
    "cobertura_aplica": true,
    "razon_cobertura": "<explicación>",
    "franquicia_aplica": false,
    "monto_sugerido_liquidar": null,
    "infracciones_detectadas": [
      {
        "infractor": "<ASEGURADO | TERCERO_1>",
        "descripcion_infraccion": "<descripción>",
        "articulo_ley_5016": "<Art. N° o null>",
        "articulo_ordenanza": "<Art. N° y ordenanza o null>"
      }
    ],
    "analisis_narrativo": "<párrafos en español formal y jurídico>",
    "alertas_liquidador": [],
    "confianza_dictamen": 0.0
  }
}
""".strip()


def ejecutar_paso_c(
    json_paso_b: dict,
    contexto_legal: str | None = None,
    precedentes: list | None = None,
) -> dict | None:
    """
    Genera el dictamen sugerido.
    Si contexto_legal se provee (modo RAG), se usa en lugar del texto integrado.
    """
    municipio = json_paso_b.get("siniestro", {}).get("municipio", "San Lorenzo")

    if contexto_legal:
        bloque_normativo = (
            f"NORMATIVA RECUPERADA SEMÁNTICAMENTE (municipio: {municipio}):\n{contexto_legal}"
        )
    else:
        bloque_normativo = (
            f"ORDENANZA MUNICIPAL — {municipio}:\n{ORDENANZA_SAN_LORENZO}\n\n"
            f"LEY N° 5016/14 — ARTÍCULOS RELEVANTES:\n{LEY_5016_FRAGMENTOS}"
        )

    bloque_precedentes = ""
    if precedentes:
        items = "\n".join(
            f"  - [{p['caso_id']}] sim={p['similitud']:.2f} → {p['responsabilidad']}"
            for p in precedentes
        )
        bloque_precedentes = f"\nPRECEDENTES SIMILARES:\n{items}\n"

    prompt = f"""
Analiza el siguiente caso de siniestro vehicular y emite una sugerencia de dictamen.

DATOS DEL CASO (Paso B):
{__import__('json').dumps(json_paso_b, ensure_ascii=False, indent=2)[:8000]}

{bloque_normativo}
{bloque_precedentes}
Devuelve EXCLUSIVAMENTE este JSON:
{_SCHEMA_C}""".strip()

    respuesta = llamar_gemini(_SYSTEM_C, prompt)
    if respuesta:
        result = parsear_json(respuesta)
        if result:
            resp = result.get("dictamen", {}).get("responsabilidad_sugerida", "?")
            logger.info(f"Paso C — municipio: {municipio} | responsabilidad: {resp}")
        return result
    return None


# ---------------------------------------------------------------------------
# Ejecutor principal
# ---------------------------------------------------------------------------

def preparar_documentos(archivos: list[dict]) -> list[dict]:
    """
    Prepara la lista de documentos para el pipeline.

    archivos: [{"ruta": str, "nombre": str}]
    Extrae texto y sube a Gemini File API los archivos que soporten OCR.
    """
    documentos = []
    for arch in archivos:
        ruta = arch["ruta"]
        nombre = arch["nombre"]
        formato = get_formato(nombre)
        ext = Path(nombre).suffix.lower()

        # Extraer texto (para texto plano y DOCX)
        contenido = extraer_texto(ruta, nombre)

        # Subir a Gemini File API para OCR nativo si aplica
        gemini_file = None
        if ext in EXTENSIONES_OCR:
            gemini_file = subir_archivo(ruta, nombre)

        documentos.append({
            "nombre_archivo": nombre,
            "formato": formato,
            "contenido": contenido,
            "gemini_file": gemini_file,
        })
    return documentos


def ejecutar_pipeline(
    caso_id: str,
    archivos: list[dict],
    usar_rag: bool = False,
) -> dict:
    """
    Ejecuta el pipeline completo A → B → C.

    archivos: [{"ruta": str, "nombre": str}]
    Retorna dict con: paso_a, paso_b, paso_c, error, tiempo_segundos.
    """
    import time as _time
    t0 = _time.time()

    if not archivos:
        return {"error": "No se recibieron archivos para procesar"}

    try:
        documentos = preparar_documentos(archivos)

        resultado_a = ejecutar_paso_a(documentos)
        if not resultado_a:
            return {"error": "Paso A: sin resultados de clasificación"}

        resultado_b = ejecutar_paso_b(resultado_a, caso_id)
        if not resultado_b:
            return {"paso_a": resultado_a, "error": "Paso B: fallo en extracción"}

        # RAG opcional
        contexto_legal = None
        precedentes = None
        if usar_rag:
            try:
                from vector_store import CorpusLegal, CasosHistoricos, construir_contexto_legal_rag
                municipio = resultado_b.get("siniestro", {}).get("municipio", "San Lorenzo")
                descripcion = resultado_b.get("siniestro", {}).get("descripcion_dinamica", caso_id)
                corpus = CorpusLegal()
                casos_hist = CasosHistoricos()
                contexto_legal = construir_contexto_legal_rag(descripcion, municipio, corpus)
                precedentes = casos_hist.buscar_similares(descripcion)
            except Exception as e:
                logger.warning(f"RAG no disponible, continuando sin él: {e}")

        resultado_c = ejecutar_paso_c(resultado_b, contexto_legal=contexto_legal, precedentes=precedentes)
        if not resultado_c:
            return {"paso_a": resultado_a, "paso_b": resultado_b, "error": "Paso C: fallo en dictamen"}

        # Si RAG disponible, indexar este caso en casos históricos
        if usar_rag and resultado_b and resultado_c:
            try:
                from vector_store import CasosHistoricos
                casos_hist = CasosHistoricos()
                casos_hist.indexar_caso(caso_id, resultado_b, resultado_c)
            except Exception:
                pass

        return {
            "caso_id": caso_id,
            "paso_a": resultado_a,
            "paso_b": resultado_b,
            "paso_c": resultado_c,
            "error": None,
            "tiempo_segundos": round(_time.time() - t0, 2),
        }

    except Exception as e:
        logger.exception(f"Error inesperado en pipeline para caso {caso_id}: {e}")
        return {"error": str(e), "tiempo_segundos": round(_time.time() - t0, 2)}
