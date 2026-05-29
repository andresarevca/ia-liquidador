"""
validar_pipeline_ia.py
======================
Script de validación Fase 1 — Sistema de Liquidación de Siniestros (Paraguay)
Ejecuta los tres pasos de IA (Clasificar → Extraer → Dictamen) con un
documento ficticio integrado. Sin base de datos, sin infraestructura.

USO:
    python validar_pipeline_ia.py

    # Para probar con tu propio PDF:
    python validar_pipeline_ia.py --pdf ruta/al/documento.pdf

OBTENER API KEY:
    https://aistudio.google.com/app/apikey (gratuito con límites generosos)
"""

import os
import sys
import json
import time
import argparse
import textwrap
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Dependencias opcionales — el script las importa con mensajes claros
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv no instalado; se usará variable de entorno directamente

try:
    import google.generativeai as genai
except ImportError:
    print("\n[ERROR] Falta instalar: pip install google-generativeai python-dotenv colorama\n")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Importaciones desde services/ (pipeline refactorizado)
# ---------------------------------------------------------------------------
try:
    from services import gemini_client as _gemini_client
    from services.pipeline import (
        ejecutar_pipeline as _ejecutar_pipeline,
        preparar_documentos as _preparar_documentos,
    )
    _SERVICES_OK = True
except ImportError:
    _SERVICES_OK = False  # Fallback a implementación local (modo legado)

# Vector store + embeddings (opcional — requiere: pip install chromadb)
try:
    from vector_store import (
        CorpusLegal,
        CasosHistoricos,
        construir_contexto_legal_rag,
    )
    _RAG_DISPONIBLE = True
except ImportError:
    _RAG_DISPONIBLE = False

try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    COLOR = True
except ImportError:
    COLOR = False
    class Fore:
        GREEN = YELLOW = RED = CYAN = MAGENTA = WHITE = ""
    class Style:
        BRIGHT = RESET_ALL = DIM = ""


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
GEMINI_MODEL_VISION = "gemini-2.5-flash"   # más permisivo en el tier gratuito
GEMINI_MODEL_FAST   = "gemini-2.5-flash"
MAX_RETRIES          = 3
RETRY_DELAY_SECONDS  = 5


# ---------------------------------------------------------------------------
# Documento ficticio integrado (simula el texto extraído de un PDF/imagen)
# ---------------------------------------------------------------------------
DOCUMENTO_FICTICIO = {
    "tipo": "CONJUNTO_DE_DOCUMENTOS",
    "caso_id": "CASO-TEST-2025-001",
    "aseguradora": "Aseguradora del Paraguay S.A.",
    "documentos": [
        {
            "nombre_archivo": "denuncia_policial.pdf",
            "formato": "PDF_ESCANEADO",
            "contenido": """
POLICÍA NACIONAL DEL PARAGUAY
COMISARÍA N° 12 — SAN LORENZO
ACTA DE DENUNCIA DE ACCIDENTE DE TRÁNSITO

Fecha: 14 de marzo de 2025
Hora: 16:45 hs.
Lugar: Intersección Av. Mariscal López y Calle Ytororó, Barrio San Francisco, San Lorenzo

CONDUCTOR 1 (Asegurado):
Nombre: Carlos Javier Rodríguez Gómez
C.I.: 4.567.890
Licencia N°: 0045678 — Categoría B
Vehículo: Toyota Hilux 2021, Blanco, Matrícula BCDE 123

CONDUCTOR 2 (Tercero):
Nombre: Ana María Fernández
C.I.: 5.123.456
Licencia: No presentó
Vehículo: Honda Civic 2018, Gris, Matrícula AFGH 456

RELATO DEL HECHO:
El vehículo del Conductor 1 circulaba por Av. Mariscal López en sentido norte.
Al llegar a la intersección con Calle Ytororó, fue impactado en el lateral
derecho por el vehículo del Conductor 2, quien no respetó la señal de PARE
ubicada en dicha esquina. Condición climática: lluvia leve.

Testigo: Juan Pérez, C.I. 3.211.000. Declaró que el vehículo gris no detuvo.

Firmado: Oficial Marcos Torres — Badge 4521
""",
        },
        {
            "nombre_archivo": "poliza_seguro.pdf",
            "formato": "PDF_DIGITAL",
            "contenido": """
ASEGURADORA DEL PARAGUAY S.A.
PÓLIZA DE SEGURO AUTOMOTOR
Número de Póliza: POL-2024-087654

Asegurado: Carlos Javier Rodríguez Gómez — C.I. 4.567.890
Vehículo: Toyota Hilux 2021 — Matrícula BCDE 123
Tipo de Cobertura: TODO RIESGO
Suma Asegurada: Gs. 180.000.000
Franquicia: Gs. 3.500.000

Vigencia: 01/08/2024 al 01/08/2025

Coberturas incluidas:
- Daños propios (colisión, vuelco, robo)
- Responsabilidad Civil ante terceros
- Asistencia en ruta 24hs

Exclusiones relevantes: conducción bajo efecto de alcohol,
uso del vehículo para competencias.

Firmado: Gerencia Técnica — Aseguradora del Paraguay S.A.
""",
        },
        {
            "nombre_archivo": "pericia_tecnica.pdf",
            "formato": "PDF_ESCANEADO",
            "contenido": """
INFORME DE PERITAJE TÉCNICO
Perito: Ing. Roberto Silva — Matrícula CIPPY 8834
Fecha de inspección: 15 de marzo de 2025
Vehículo inspeccionado: Toyota Hilux 2021 — BCDE 123

DAÑOS CONSTATADOS:
- Puerta trasera derecha: deformación estructural severa, requiere reemplazo
- Estribo derecho: doblado, requiere reemplazo
- Guardabarro trasero derecho: deformado, requiere reemplazo
- Umbral lateral derecho: deformación leve, requiere enderezado

Los daños son consistentes con un impacto lateral de intensidad media-alta
en desplazamiento a velocidad urbana (~40 km/h).

VALORACIÓN DE DAÑOS:
Mano de obra:        Gs.  8.500.000
Repuestos originales: Gs. 20.000.000
TOTAL ESTIMADO:      Gs. 28.500.000

Nota: El sello del perito presenta tinta corrida por humedad en copia adjunta.
""",
        },
        {
            "nombre_archivo": "denuncia_aseguradora.pdf",
            "formato": "PDF_DIGITAL",
            "contenido": """
FORMULARIO DE DENUNCIA DE SINIESTRO
Aseguradora del Paraguay S.A.

Número de Póliza: POL-2024-087654
Fecha del siniestro: 14/03/2025
Hora declarada: 17:10 hs.     <-- NOTA: difiere 25 min de denuncia policial
Lugar: Av. Mariscal López esq. Ytororó, San Lorenzo

Descripción del asegurado:
Circulaba normalmente cuando fui impactado por un vehículo que no respetó el pare.
No consumí alcohol ni sustancias. Solicito cobertura por daños propios.

Firmado: Carlos J. Rodríguez G.        Fecha recepción: 15/03/2025
""",
        },
        {
            "nombre_archivo": "foto_evento_01.jpg",
            "formato": "IMAGEN",
            "contenido": """
[DESCRIPCIÓN DE IMAGEN — simulada para texto]
Fotografía tomada en escena. Se observan dos vehículos en intersección urbana.
Toyota Hilux blanco con daño visible en lateral derecho.
Honda Civic gris con daño en frente. Señal de PARE visible en poste esquinero.
Asfalto húmedo. Presencia de funcionarios policiales.
""",
        },
        {
            "nombre_archivo": "cedula_verde.pdf",
            "formato": "PDF_DIGITAL",
            "contenido": """
REGISTRO AUTOMOTOR DEL PARAGUAY
TÍTULO DE PROPIEDAD — CÉDULA VERDE

Propietario: Carlos Javier Rodríguez Gómez — C.I. 4.567.890
Marca/Modelo: Toyota Hilux 4x4 SR 2021
Matrícula: BCDE 123
Chasis: JTFHX02P900123456
Motor: 2GD-FTV
Color: Blanco
Fecha de inscripción: 10/09/2021
""",
        },
    ],
}

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
"""

LEY_5016_FRAGMENTOS = """
LEY N° 5016/14 — DE TRÁNSITO Y SEGURIDAD VIAL — REPÚBLICA DEL PARAGUAY

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

Art. 89 — CONDICIONES CLIMÁTICAS:
En condiciones de lluvia, el conductor deberá reducir la velocidad y aumentar
la distancia de seguridad respecto al vehículo precedente.
"""


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def titulo(texto, color=Fore.CYAN):
    linea = "─" * 60
    print(f"\n{color}{Style.BRIGHT}{linea}")
    print(f"  {texto}")
    print(f"{linea}{Style.RESET_ALL}")


def ok(texto):
    print(f"{Fore.GREEN}  ✓ {texto}{Style.RESET_ALL}")


def warn(texto):
    print(f"{Fore.YELLOW}  ⚠ {texto}{Style.RESET_ALL}")


def error(texto):
    print(f"{Fore.RED}  ✗ {texto}{Style.RESET_ALL}")


def info(texto):
    print(f"{Style.DIM}  {texto}{Style.RESET_ALL}")


def parsear_json_seguro(texto_crudo: str) -> dict | None:
    """
    Intenta parsear JSON de la respuesta de Gemini.
    Maneja el caso en que el modelo agregue bloques ```json ... ```.
    """
    texto = texto_crudo.strip()
    # Remover bloques de código Markdown si los hay
    if texto.startswith("```"):
        lineas = texto.split("\n")
        texto = "\n".join(lineas[1:])
        if texto.strip().endswith("```"):
            texto = texto.strip()[:-3]
    try:
        return json.loads(texto)
    except json.JSONDecodeError as e:
        error(f"JSON inválido: {e}")
        info(f"Primeros 300 chars de la respuesta:\n{texto_crudo[:300]}")
        return None


def llamar_gemini(modelo: str, prompt_sistema: str, prompt_usuario: str,
                  partes: list | None = None,
                  reintentos: int = MAX_RETRIES) -> str | None:
    """
    Llama a Gemini con reintentos automáticos ante errores de red o rate limit.
    Si se proveen 'partes' (archivos Gemini File u otros fragmentos de texto),
    se construye contenido multimodal: [*partes, prompt_usuario].
    Esto permite usar el OCR nativo de Gemini para PDFs e imágenes.
    """
    model = genai.GenerativeModel(
        model_name=modelo,
        system_instruction=prompt_sistema,
    )
    contenido = (list(partes) + [prompt_usuario]) if partes else prompt_usuario
    for intento in range(1, reintentos + 1):
        try:
            respuesta = model.generate_content(contenido)
            return respuesta.text
        except Exception as e:
            tipo_error = type(e).__name__
            if intento < reintentos:
                warn(f"Error en intento {intento}/{reintentos} ({tipo_error}). Reintentando en {RETRY_DELAY_SECONDS}s...")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                error(f"Falló después de {reintentos} intentos: {e}")
                return None


# ---------------------------------------------------------------------------
# PASO A — Clasificador de documentos
# ---------------------------------------------------------------------------

SYSTEM_A = """
Eres un clasificador especializado en documentos de siniestros vehiculares del Paraguay.
Tu única tarea es analizar el contenido de cada archivo adjunto y devolver
una clasificación estructurada en JSON.

REGLAS ESTRICTAS:
1. Devuelve ÚNICAMENTE el objeto JSON. Sin texto adicional, sin Markdown, sin explicaciones.
2. Si el documento es ilegible o de baja calidad, marca legible: false.
3. No extraigas datos del contenido en este paso. Solo clasifica.
4. Ante la duda entre dos tipos, usa OTRO y describe en nota_clasificacion.
""".strip()

def prompt_paso_a(doc: dict, con_ocr: bool = False) -> str:
    if con_ocr:
        bloque_contenido = (
            "El documento ha sido adjuntado directamente para su análisis con OCR nativo."
        )
    else:
        bloque_contenido = f"CONTENIDO DEL DOCUMENTO:\n{doc['contenido']}"

    return f"""
Analiza el siguiente documento adjunto del caso de siniestro vehicular.

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
  "pertenece_al_caso": <true | false | "indefinido">,
  "razon_pertenencia": "<una frase explicando por qué>"
}}
""".strip()


def ejecutar_paso_a(documentos: list) -> list:
    titulo("PASO A — Clasificación de documentos", Fore.YELLOW)
    resultados = []
    for doc in documentos:
        gemini_file = doc.get("gemini_file")
        modo = "OCR Gemini" if gemini_file else "texto plano"
        info(f"Clasificando: {doc['nombre_archivo']} ({doc['formato']}) [{modo}]...")
        partes = [gemini_file] if gemini_file else None
        respuesta = llamar_gemini(
            GEMINI_MODEL_VISION, SYSTEM_A,
            prompt_paso_a(doc, con_ocr=bool(gemini_file)),
            partes=partes,
        )
        if respuesta is None:
            error(f"Sin respuesta para {doc['nombre_archivo']}")
            resultados.append({"archivo": doc["nombre_archivo"], "error": True})
            continue
        parsed = parsear_json_seguro(respuesta)
        if parsed:
            parsed["archivo"] = doc["nombre_archivo"]
            parsed["contenido_original"] = doc.get("contenido", "")
            parsed["gemini_file"] = gemini_file  # propagar para el Paso B
            resultados.append(parsed)
            confianza = parsed.get("confianza", 0)
            tipo = parsed.get("tipo_doc", "?")
            legible = parsed.get("legible", False)
            estado = ok if legible and confianza > 0.7 else warn
            estado(f"{doc['nombre_archivo']} → {tipo} (confianza: {confianza:.2f})")
            if parsed.get("nota_clasificacion"):
                info(f"   Nota: {parsed['nota_clasificacion']}")
        else:
            resultados.append({"archivo": doc["nombre_archivo"], "error": True, "raw": respuesta})
    return resultados


# ---------------------------------------------------------------------------
# PASO B — Extractor de variables críticas
# ---------------------------------------------------------------------------

SYSTEM_B = """
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

# Schema compartido entre el modo texto y el modo OCR
_PROMPT_PASO_B_SCHEMA = """
MUNICIPIOS VÁLIDOS: Asunción | San Lorenzo | Luque | Capiatá | Lambaré | Fernando de la Mora |
Ciudad del Este | Encarnación | Coronel Oviedo | Villarrica | Otro

Devuelve EXCLUSIVAMENTE este JSON:

{
  "siniestro": {
    "fecha_hora": "<ISO 8601 o null>",
    "municipio": "<municipio de la lista o null>",
    "direccion_aproximada": "<calle, barrio o null>",
    "tipo_via": "<CALLE_URBANA | RUTA_NACIONAL | RUTA_DEPARTAMENTAL | ESTACIONAMIENTO | OTRA | null>",
    "condicion_climatica": "<DESPEJADO | LLUVIA | NIEBLA | NOCHE | DESCONOCIDO>",
    "descripcion_dinamica": "<párrafo descriptivo en español formal>"
  },
  "vehiculos": [
    {
      "rol": "<ASEGURADO | TERCERO_1>",
      "marca": "<o null>",
      "modelo": "<o null>",
      "año": "<número o null>",
      "matricula": "<o null>",
      "color": "<o null>",
      "conductor_nombre": "<o null>",
      "conductor_ci": "<o null>",
      "licencia_numero": "<o null>",
      "licencia_categoria": "<o null>",
      "danios_descripcion": "<o null>"
    }
  ],
  "poliza": {
    "numero": "<o null>",
    "aseguradora": "<o null>",
    "vigencia_desde": "<ISO 8601 o null>",
    "vigencia_hasta": "<ISO 8601 o null>",
    "cobertura_tipo": "<RESPONSABILIDAD_CIVIL | TODO_RIESGO | PARCIAL | DESCONOCIDO>",
    "suma_asegurada": "<número PYG o null>",
    "franquicia": "<número PYG o null>"
  },
  "documentacion": {
    "poliza_presente": "<true | false>",
    "denuncia_policial_presente": "<true | false>",
    "denuncia_administrativa_presente": "<true | false>",
    "pericia_tecnica_presente": "<true | false>",
    "fotos_evento_cantidad": "<número>",
    "cedula_verde_presente": "<true | false>",
    "licencia_conductor_presente": "<true | false>"
  },
  "monto_danios": {
    "estimacion_pericia": "<número PYG o null>",
    "moneda_original": "PYG"
  },
  "conflictos": [
    {
      "campo": "<nombre del campo>",
      "valor_doc_1": "<valor según documento A>",
      "valor_doc_2": "<valor según documento B>",
      "descripcion": "<explicación>"
    }
  ],
  "calidad_extraccion": {
    "score": "<0.0 a 1.0>",
    "campos_faltantes_criticos": ["<lista>"],
    "observaciones": "<nota general o null>"
  }
}
""".strip()


def prompt_paso_b(resultados_a: list, caso_id: str) -> str:
    """Modo texto: incrusta el contenido de todos los documentos en el prompt."""
    bloques = []
    for r in resultados_a:
        if r.get("error"):
            continue
        tipo = r.get("tipo_doc", "DESCONOCIDO")
        contenido = r.get("contenido_original", "")
        bloques.append(f"--- DOCUMENTO: {r['archivo']} | TIPO: {tipo} ---\n{contenido}")
    documentos_texto = "\n\n".join(bloques)

    return f"""
Analiza los siguientes documentos del caso {caso_id} y extrae todas las variables estructuradas.

DOCUMENTOS DEL CASO:
{documentos_texto}

{_PROMPT_PASO_B_SCHEMA}
""".strip()


def _prompt_paso_b_ocr(caso_id: str) -> str:
    """Modo OCR: los documentos vienen como partes de archivo adjuntas, no como texto."""
    return f"""
Analiza los documentos del caso {caso_id} adjuntos arriba (cada uno precedido de su etiqueta)
y extrae todas las variables estructuradas.

{_PROMPT_PASO_B_SCHEMA}
""".strip()


def ejecutar_paso_b(resultados_a: list, caso_id: str) -> dict | None:
    titulo("PASO B — Extracción de variables críticas", Fore.CYAN)

    # Verificar si hay archivos subidos a la File API
    hay_archivos_ocr = any(
        r.get("gemini_file") for r in resultados_a if not r.get("error")
    )

    if hay_archivos_ocr:
        info("Modo OCR Gemini: construyendo contenido multimodal [archivo + etiqueta]...")
        partes = []
        for r in resultados_a:
            if r.get("error"):
                continue
            tipo = r.get("tipo_doc", "DESCONOCIDO")
            partes.append(f"--- DOCUMENTO: {r['archivo']} | TIPO: {tipo} ---")
            if r.get("gemini_file"):
                partes.append(r["gemini_file"])  # objeto File — Gemini lee el doc con OCR
            else:
                partes.append(r.get("contenido_original", ""))  # fallback texto
        respuesta = llamar_gemini(
            GEMINI_MODEL_FAST, SYSTEM_B,
            _prompt_paso_b_ocr(caso_id),
            partes=partes,
        )
    else:
        info("Modo texto: enviando todos los documentos clasificados a Gemini...")
        respuesta = llamar_gemini(GEMINI_MODEL_FAST, SYSTEM_B, prompt_paso_b(resultados_a, caso_id))

    if respuesta is None:
        return None
    parsed = parsear_json_seguro(respuesta)
    if parsed:
        municipio = parsed.get("siniestro", {}).get("municipio", "?")
        fecha = parsed.get("siniestro", {}).get("fecha_hora", "?")
        cobertura = parsed.get("poliza", {}).get("cobertura_tipo", "?")
        score = parsed.get("calidad_extraccion", {}).get("score", "?")
        conflictos = parsed.get("conflictos", [])
        ok(f"Municipio: {municipio}")
        ok(f"Fecha/hora siniestro: {fecha}")
        ok(f"Cobertura: {cobertura}")
        ok(f"Score de extracción: {score}")
        if conflictos:
            warn(f"Se detectaron {len(conflictos)} conflicto(s) entre documentos:")
            for c in conflictos:
                info(f"   [{c.get('campo')}] {c.get('descripcion')}")
        faltantes = parsed.get("calidad_extraccion", {}).get("campos_faltantes_criticos", [])
        if faltantes:
            warn(f"Campos faltantes críticos: {', '.join(faltantes)}")
    return parsed


# ---------------------------------------------------------------------------
# PASO C — Análisis normativo y dictamen sugerido
# ---------------------------------------------------------------------------

SYSTEM_C = """
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

def prompt_paso_c(
    json_paso_b: dict,
    municipio: str,
    contexto_legal: str | None = None,
) -> str:
    """
    Construye el prompt del Paso C.
    Si contexto_legal es provisto (modo RAG), lo usa en lugar de los textos
    legales hardcodeados.
    """
    if contexto_legal is None:
        bloque_normativo = (
            f"ORDENANZA MUNICIPAL — {municipio}:\n{ORDENANZA_SAN_LORENZO}\n\n"
            f"LEY N° 5016/14 — ARTÍCULOS RELEVANTES:\n{LEY_5016_FRAGMENTOS}"
        )
    else:
        bloque_normativo = (
            f"NORMATIVA RECUPERADA SEMÁNTICAMENTE (municipio: {municipio}):\n"
            f"{contexto_legal}"
        )
    return f"""
Analiza el siguiente caso de siniestro vehicular y emite una sugerencia de dictamen.

DATOS DEL CASO (Paso B):
{json.dumps(json_paso_b, ensure_ascii=False, indent=2)}

{bloque_normativo}

Devuelve EXCLUSIVAMENTE este JSON:

{{
  "dictamen": {{
    "dictamen_posible": "<true | false>",
    "datos_faltantes_para_dictamen": ["<lista si dictamen_posible es false>"],
    "responsabilidad_sugerida": "<ASEGURADO_RESPONSABLE | TERCERO_RESPONSABLE | RESPONSABILIDAD_COMPARTIDA | CASO_FORTUITO | INDETERMINADO>",
    "porcentaje_responsabilidad_asegurado": "<0-100 o null>",
    "porcentaje_responsabilidad_tercero": "<0-100 o null>",
    "cobertura_aplica": "<true | false | condicional>",
    "razon_cobertura": "<explicación>",
    "franquicia_aplica": "<true | false>",
    "monto_sugerido_liquidar": "<número PYG o null>",
    "infracciones_detectadas": [
      {{
        "infractor": "<ASEGURADO | TERCERO_1>",
        "descripcion_infraccion": "<descripción>",
        "articulo_ley_5016": "<Art. N° o null>",
        "articulo_ordenanza": "<Art. N° y ordenanza o null>"
      }}
    ],
    "analisis_narrativo": "<párrafos en español formal y jurídico>",
    "alertas_liquidador": ["<lista de observaciones críticas>"],
    "confianza_dictamen": "<0.0 a 1.0>"
  }}
}}
""".strip()


def ejecutar_paso_c(
    json_paso_b: dict,
    contexto_legal: str | None = None,
    precedentes: list | None = None,
) -> dict | None:
    titulo("PASO C — Análisis normativo y dictamen sugerido", Fore.MAGENTA)
    municipio = json_paso_b.get("siniestro", {}).get("municipio", "San Lorenzo")
    info(f"Aplicando normativa de: {municipio}")
    if contexto_legal is not None:
        info("Modo RAG: usando artículos recuperados semánticamente")
    if precedentes:
        info(f"Precedentes similares encontrados: {len(precedentes)}")
        for p in precedentes:
            info(f"   [{p['caso_id']}] sim={p['similitud']:.2f} → {p['responsabilidad']}")
    respuesta = llamar_gemini(
        GEMINI_MODEL_FAST, SYSTEM_C,
        prompt_paso_c(json_paso_b, municipio, contexto_legal=contexto_legal),
    )
    if respuesta is None:
        return None
    parsed = parsear_json_seguro(respuesta)
    if parsed:
        d = parsed.get("dictamen", {})
        responsabilidad = d.get("responsabilidad_sugerida", "?")
        cobertura = d.get("cobertura_aplica", "?")
        monto = d.get("monto_sugerido_liquidar", "?")
        confianza = d.get("confianza_dictamen", "?")
        infracciones = d.get("infracciones_detectadas", [])
        alertas = d.get("alertas_liquidador", [])

        ok(f"Responsabilidad sugerida: {responsabilidad}")
        ok(f"Cobertura aplica: {cobertura}")
        if monto:
            monto_fmt = f"Gs. {int(monto):,}".replace(",", ".") if isinstance(monto, (int, float)) else monto
            ok(f"Monto sugerido a liquidar: {monto_fmt}")
        ok(f"Confianza del dictamen: {confianza}")

        if infracciones:
            info(f"\n  Infracciones detectadas ({len(infracciones)}):")
            for inf in infracciones:
                info(f"   [{inf.get('infractor')}] {inf.get('descripcion_infraccion')}")
                if inf.get("articulo_ley_5016"):
                    info(f"     → {inf.get('articulo_ley_5016')} Ley 5016/14")
                if inf.get("articulo_ordenanza"):
                    info(f"     → {inf.get('articulo_ordenanza')}")

        if alertas:
            warn(f"\n  Alertas para el liquidador ({len(alertas)}):")
            for alerta in alertas:
                warn(f"   • {alerta}")
    return parsed


# ---------------------------------------------------------------------------
# Reporte final
# ---------------------------------------------------------------------------

def imprimir_reporte(resultados_a, json_b, json_c, tiempo_total):
    titulo("REPORTE FINAL DEL PIPELINE", Fore.GREEN)

    # Documentos clasificados
    total_docs = len(resultados_a)
    docs_ok = sum(1 for r in resultados_a if not r.get("error") and r.get("legible"))
    docs_error = sum(1 for r in resultados_a if r.get("error"))
    print(f"\n  Documentos procesados:  {total_docs}")
    print(f"  Legibles y clasificados: {Fore.GREEN}{docs_ok}{Style.RESET_ALL}")
    if docs_error:
        print(f"  Con error:              {Fore.RED}{docs_error}{Style.RESET_ALL}")

    # Calidad extracción Paso B
    if json_b:
        score = json_b.get("calidad_extraccion", {}).get("score", 0)
        score_float = float(score) if isinstance(score, (int, float, str)) else 0
        color_score = Fore.GREEN if score_float >= 0.8 else (Fore.YELLOW if score_float >= 0.6 else Fore.RED)
        print(f"\n  Score extracción (Paso B): {color_score}{score_float:.2f}{Style.RESET_ALL}")
        conflictos = json_b.get("conflictos", [])
        if conflictos:
            print(f"  Conflictos entre docs:    {Fore.YELLOW}{len(conflictos)}{Style.RESET_ALL}")

    # Dictamen Paso C
    if json_c:
        d = json_c.get("dictamen", {})
        responsabilidad = d.get("responsabilidad_sugerida", "?")
        confianza = d.get("confianza_dictamen", 0)
        confianza_float = float(confianza) if isinstance(confianza, (int, float, str)) else 0
        color_conf = Fore.GREEN if confianza_float >= 0.8 else (Fore.YELLOW if confianza_float >= 0.6 else Fore.RED)
        print(f"\n  Responsabilidad sugerida:  {Fore.CYAN}{responsabilidad}{Style.RESET_ALL}")
        print(f"  Confianza dictamen:        {color_conf}{confianza_float:.2f}{Style.RESET_ALL}")
        monto = d.get("monto_sugerido_liquidar")
        if monto:
            monto_fmt = f"Gs. {int(monto):,}".replace(",", ".") if isinstance(monto, (int, float)) else monto
            print(f"  Monto sugerido:            {Fore.GREEN}{monto_fmt}{Style.RESET_ALL}")

    print(f"\n  Tiempo total:  {tiempo_total:.1f} segundos")

    # Veredicto de validación
    pipeline_ok = (
        json_b is not None and
        json_c is not None and
        json_c.get("dictamen", {}).get("dictamen_posible") in [True, "true"]
    )
    print()
    if pipeline_ok:
        print(f"{Fore.GREEN}{Style.BRIGHT}  ✓ PIPELINE VALIDADO — listo para avanzar a Fase 2{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}{Style.BRIGHT}  ⚠ PIPELINE INCOMPLETO — revisar los errores anteriores{Style.RESET_ALL}")
    print()


def guardar_resultados(resultados_a, json_b, json_c, caso_id):
    """Guarda los JSON de cada paso en resultados/<caso_id>_<timestamp>/."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path("resultados") / f"{caso_id}_{timestamp}"
    base.mkdir(parents=True, exist_ok=True)

    resultados_a_serializables = [
        {k: v for k, v in r.items() if k != "gemini_file"}
        for r in resultados_a
    ]
    with open(base / "paso_a_clasificacion.json", "w", encoding="utf-8") as f:
        json.dump(resultados_a_serializables, f, ensure_ascii=False, indent=2)
    if json_b:
        with open(base / "paso_b_extraccion.json", "w", encoding="utf-8") as f:
            json.dump(json_b, f, ensure_ascii=False, indent=2)
    if json_c:
        with open(base / "paso_c_dictamen.json", "w", encoding="utf-8") as f:
            json.dump(json_c, f, ensure_ascii=False, indent=2)

    ok(f"Resultados guardados en: {base}/")
    return base


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Carga de documentos desde carpeta
# ---------------------------------------------------------------------------

EXTENSIONES_SOPORTADAS = {
    ".pdf":  "PDF_ESCANEADO",
    ".jpg":  "IMAGEN",
    ".jpeg": "IMAGEN",
    ".png":  "IMAGEN",
    ".gif":  "IMAGEN",
    ".webp": "IMAGEN",
    ".txt":  "TEXTO_PLANO",
    ".docx": "WORD_DOC",
    ".doc":  "WORD_DOC",
    ".odt":  "WORD_DOC",
}

# MIME types para la File API de Gemini (OCR nativo)
MIME_TYPES = {
    ".pdf":  "application/pdf",
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".gif":  "image/gif",
    ".webp": "image/webp",
}


def subir_archivo_a_gemini(archivo: Path):
    """
    Sube un archivo a la File API de Gemini y devuelve el objeto File.
    Gemini procesa el contenido directamente (OCR para PDFs e imágenes).
    Los archivos subidos se mantienen disponibles 48 horas.
    Retorna None si el formato no es soportado o si falla la subida.
    """
    mime = MIME_TYPES.get(archivo.suffix.lower())
    if not mime:
        return None  # .txt no necesita subirse, se envía como texto plano
    try:
        info(f"   Subiendo a Gemini File API: {archivo.name} ({mime})...")
        gemini_file = genai.upload_file(path=str(archivo), mime_type=mime,
                                        display_name=archivo.name)
        ok(f"   OCR listo: {archivo.name}")
        return gemini_file
    except Exception as e:
        warn(f"   No se pudo subir {archivo.name} a la File API: {e}")
        return None


def extraer_contenido_archivo(archivo: Path, formato: str) -> str:
    """Extrae el contenido textual de un archivo según su formato."""
    if formato == "TEXTO_PLANO":
        return archivo.read_text(encoding="utf-8", errors="replace")

    if formato == "PDF_ESCANEADO":
        # Intentar con pdfplumber (preferido)
        try:
            import pdfplumber
            with pdfplumber.open(archivo) as pdf:
                texto = "\n".join(p.extract_text() or "" for p in pdf.pages).strip()
            if texto:
                return texto
        except ImportError:
            pass
        except Exception as e:
            warn(f"pdfplumber falló en {archivo.name}: {e}")

        # Fallback: PyPDF2
        try:
            import PyPDF2
            with open(archivo, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                texto = "\n".join(
                    (p.extract_text() or "") for p in reader.pages
                ).strip()
            if texto:
                return texto
        except ImportError:
            pass
        except Exception as e:
            warn(f"PyPDF2 falló en {archivo.name}: {e}")

        # Sin extractor disponible
        return (
            f"[PDF sin texto extraíble: {archivo.name}]\n"
            "Instalar 'pdfplumber' o 'PyPDF2' para extracción automática.\n"
            "En Fase 2 Gemini procesará el archivo directamente vía API multimodal."
        )

    if formato == "IMAGEN":
        return (
            f"[Imagen real: {archivo.name}]\n"
            "En Fase 2 Gemini procesará la imagen directamente vía API multimodal."
        )

    if formato == "WORD_DOC":
        sufijo = archivo.suffix.lower()
        if sufijo == ".docx":
            try:
                import docx
                doc = docx.Document(str(archivo))
                texto = "\n".join(p.text for p in doc.paragraphs).strip()
                if texto:
                    return texto
            except ImportError:
                pass
            except Exception as e:
                warn(f"python-docx falló en {archivo.name}: {e}")
            return (
                f"[DOCX sin texto extraíble: {archivo.name}]\n"
                "Instalar 'python-docx' para extracción automática."
            )
        if sufijo == ".odt":
            try:
                from odf.opendocument import load as odf_load
                from odf.text import P
                odf_doc = odf_load(str(archivo))
                texto = "\n".join(
                    str(el) for el in odf_doc.getElementsByType(P)
                ).strip()
                if texto:
                    return texto
            except ImportError:
                pass
            except Exception as e:
                warn(f"odfpy falló en {archivo.name}: {e}")
            return (
                f"[ODT sin texto extraíble: {archivo.name}]\n"
                "Instalar 'odfpy' para extracción automática."
            )
        # .doc binario — requiere herramienta externa (antiword / LibreOffice)
        return (
            f"[DOC binario: {archivo.name}]\n"
            "El formato .doc antiguo no es soportado directamente. "
            "Convertir a .docx con LibreOffice antes de procesar."
        )

    return f"[Formato no soportado para extracción de texto: {archivo.name}]"


def cargar_documentos_de_carpeta(ruta_carpeta: Path) -> list:
    """Lee todos los archivos soportados de una carpeta y los convierte al formato interno."""
    if not ruta_carpeta.is_dir():
        error(f"La ruta no es una carpeta válida: {ruta_carpeta}")
        sys.exit(1)

    archivos = sorted(
        f for f in ruta_carpeta.iterdir()
        if f.is_file() and f.suffix.lower() in EXTENSIONES_SOPORTADAS
    )

    if not archivos:
        error(
            f"No se encontraron archivos soportados en: {ruta_carpeta}\n"
            f"  Extensiones aceptadas: {', '.join(EXTENSIONES_SOPORTADAS)}"
        )
        sys.exit(1)

    ok(f"Se encontraron {len(archivos)} archivo(s) en '{ruta_carpeta}':")
    titulo("SUBIENDO ARCHIVOS A GEMINI (OCR nativo)", Fore.YELLOW)
    documentos = []
    for archivo in archivos:
        formato = EXTENSIONES_SOPORTADAS[archivo.suffix.lower()]
        info(f"   {archivo.name}  [{formato}]")
        # Archivos de texto: leer directamente; PDFs e imágenes: subir a la File API
        if formato == "TEXTO_PLANO":
            contenido = archivo.read_text(encoding="utf-8", errors="replace")
            gemini_file = None
        elif formato == "WORD_DOC":
            contenido = extraer_contenido_archivo(archivo, formato)
            gemini_file = None
        else:
            contenido = f"[OCR Gemini: {archivo.name}]"
            gemini_file = subir_archivo_a_gemini(archivo)
        documentos.append({
            "nombre_archivo": archivo.name,
            "formato": formato,
            "contenido": contenido,
            "gemini_file": gemini_file,
        })
    return documentos


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Validación del pipeline de IA para liquidación de siniestros"
    )
    parser.add_argument(
        "--pdf", type=str, default=None,
        help="Ruta a un PDF real para usar en lugar del documento ficticio integrado"
    )
    parser.add_argument(
        "--carpeta", type=str, default=None,
        help="Ruta a una carpeta con los documentos del caso (PDFs, imágenes, .txt). "
             "Reemplaza el caso ficticio integrado. Ej: --carpeta carpeta/"
    )
    parser.add_argument(
        "--solo-paso", type=int, choices=[1, 2, 3], default=None,
        help="Ejecutar solo el paso indicado (útil para debug)"
    )
    parser.add_argument(
        "--rag", action="store_true",
        help="Activar RAG: indexa el corpus legal y busca artículos relevantes "
             "semánticamente para el Paso C (requiere: pip install chromadb)"
    )
    parser.add_argument(
        "--indexar-ordenanza", type=str, default=None, metavar="PDF",
        help="Ruta a un PDF de ordenanza municipal para indexar en el corpus legal "
             "(implica --rag). Ej: --indexar-ordenanza carpeta/Ordenanza.pdf "
             "--municipio-ordenanza Katueté"
    )
    parser.add_argument(
        "--municipio-ordenanza", type=str, default=None,
        help="Municipio de la ordenanza a indexar (usar con --indexar-ordenanza)"
    )
    args = parser.parse_args()

    # Verificar API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        error("Variable de entorno GEMINI_API_KEY no configurada.")
        info("Exportala con: export GEMINI_API_KEY='tu_api_key'")
        info("Obtené una gratis en: https://aistudio.google.com/app/apikey")
        sys.exit(1)

    genai.configure(api_key=api_key)

    print(f"\n{Fore.CYAN}{Style.BRIGHT}")
    print("  ╔══════════════════════════════════════════════════╗")
    print("  ║   Validación Pipeline IA — Liquidador Siniestros ║")
    print("  ║   Paraguay · Ley 5016/14                         ║")
    print("  ╚══════════════════════════════════════════════════╝")
    print(f"{Style.RESET_ALL}")

    # -----------------------------------------------------------------------
    # Inicialización RAG (opcional)
    # -----------------------------------------------------------------------
    corpus_legal   = None
    casos_hist     = None
    usar_rag       = args.rag or bool(args.indexar_ordenanza)

    if usar_rag:
        if not _RAG_DISPONIBLE:
            warn("--rag requiere ChromaDB: pip install chromadb")
            warn("Continuando sin RAG...")
            usar_rag = False
        else:
            titulo("INICIALIZANDO VECTOR STORE (RAG)", Fore.CYAN)
            corpus_legal = CorpusLegal()
            casos_hist   = CasosHistoricos()

            # Indexar corpus legal integrado (idempotente)
            n_ley = corpus_legal.indexar_texto(
                "Ley 5016-14", LEY_5016_FRAGMENTOS,
                municipio="Nacional", tipo="LEY",
            )
            n_ord = corpus_legal.indexar_texto(
                "Ordenanza 45-2019 San Lorenzo", ORDENANZA_SAN_LORENZO,
                municipio="San Lorenzo", tipo="ORDENANZA",
            )
            if n_ley or n_ord:
                ok(f"Corpus legal: {n_ley} chunk(s) Ley 5016 + {n_ord} chunk(s) Ordenanza SL")
            else:
                info(f"Corpus legal ya indexado ({corpus_legal.contar()} chunks en total)")

            # Indexar PDF de ordenanza adicional si se proveyó
            if args.indexar_ordenanza:
                arch_ord = Path(args.indexar_ordenanza)
                if not arch_ord.exists():
                    error(f"Ordenanza no encontrada: {args.indexar_ordenanza}")
                else:
                    municipio_ord = args.municipio_ordenanza or arch_ord.stem
                    try:
                        n = corpus_legal.indexar_pdf(arch_ord, municipio=municipio_ord)
                        ok(f"Ordenanza indexada: {arch_ord.name} → {n} chunk(s) ({municipio_ord})")
                    except Exception as e:
                        warn(f"No se pudo indexar la ordenanza: {e}")

            # Cargar casos históricos previos (idempotente)
            n_hist = casos_hist.cargar_desde_resultados()
            if n_hist:
                ok(f"{n_hist} caso(s) histórico(s) nuevo(s) indexado(s)")
            else:
                info(f"Casos históricos: {casos_hist.contar()} en el store")

    caso_id = DOCUMENTO_FICTICIO["caso_id"]
    documentos = DOCUMENTO_FICTICIO["documentos"]

    # --carpeta: reemplaza el caso ficticio por los archivos de la carpeta
    if args.carpeta:
        titulo("CARGANDO DOCUMENTOS DESDE CARPETA", Fore.YELLOW)
        documentos = cargar_documentos_de_carpeta(Path(args.carpeta))
        caso_id = Path(args.carpeta).resolve().name  # usa el nombre de la carpeta como ID

    # --pdf: agrega un archivo individual al caso (ficticio o de carpeta)
    elif args.pdf:
        ruta = Path(args.pdf)
        if not ruta.exists():
            error(f"El archivo no existe: {args.pdf}")
            sys.exit(1)
        formato = EXTENSIONES_SOPORTADAS.get(ruta.suffix.lower(), "PDF_ESCANEADO")
        warn(f"Modo PDF real: {ruta.name} [{formato}]")
        if formato == "TEXTO_PLANO":
            contenido = ruta.read_text(encoding="utf-8", errors="replace")
            gemini_file = None
        else:
            contenido = f"[OCR Gemini: {ruta.name}]"
            titulo("SUBIENDO ARCHIVO A GEMINI (OCR nativo)", Fore.YELLOW)
            gemini_file = subir_archivo_a_gemini(ruta)
        documentos.append({
            "nombre_archivo": ruta.name,
            "formato": formato,
            "contenido": contenido,
            "gemini_file": gemini_file,
        })

    inicio = time.time()

    # -----------------------------------------------------------------------
    # Ejecución del pipeline — usa services/ si está disponible
    # -----------------------------------------------------------------------
    if _SERVICES_OK and args.solo_paso is None and args.carpeta:
        # Modo services/: delegar completamente al módulo refactorizado
        _gemini_client.configure(api_key)
        info("Ejecutando pipeline via services/ (modo microservicio)...")
        archivos = [{"ruta": str(Path(args.carpeta) / d["nombre_archivo"]), "nombre": d["nombre_archivo"]}
                    for d in documentos]
        resultado = _ejecutar_pipeline(caso_id, archivos, usar_rag=usar_rag)
        resultados_a = resultado.get("paso_a", [])
        json_b = resultado.get("paso_b")
        json_c = resultado.get("paso_c")
        tiempo_total = resultado.get("tiempo_segundos", time.time() - inicio)
    else:
        # Modo legado: ejecutar paso a paso con las funciones locales del script
        # Paso A
        if args.solo_paso is None or args.solo_paso == 1:
            resultados_a = ejecutar_paso_a(documentos)
        else:
            resultados_a = []
            warn("Paso A omitido por --solo-paso")

        # Paso B
        json_b = None
        if args.solo_paso is None or args.solo_paso == 2:
            json_b = ejecutar_paso_b(resultados_a, caso_id)
        else:
            warn("Paso B omitido por --solo-paso")

        # Paso C
        json_c = None
        if (args.solo_paso is None or args.solo_paso == 3) and json_b:
            contexto_legal_rag = None
            precedentes_rag    = None
            if usar_rag and corpus_legal:
                desc_caso = json_b.get("siniestro", {}).get("descripcion_dinamica", "")
                municipio_caso = json_b.get("siniestro", {}).get("municipio", "")
                if desc_caso:
                    contexto_legal_rag = construir_contexto_legal_rag(
                        desc_caso, municipio_caso, corpus_legal
                    )
                    if casos_hist:
                        precedentes_rag = casos_hist.buscar_similares(desc_caso)
            json_c = ejecutar_paso_c(
                json_b,
                contexto_legal=contexto_legal_rag,
                precedentes=precedentes_rag,
            )
        elif json_b is None and args.solo_paso != 1:
            warn("Paso C omitido — Paso B no produjo resultado")

        tiempo_total = time.time() - inicio

    # Reporte
    imprimir_reporte(resultados_a, json_b, json_c, tiempo_total)

    # Guardar resultados
    if resultados_a or json_b or json_c:
        carpeta = guardar_resultados(resultados_a, json_b, json_c, caso_id)

        # Indexar el caso recién procesado en el store de precedentes
        if usar_rag and casos_hist and json_b and json_c:
            timestamp = carpeta.name.split("_", 1)[-1] if "_" in carpeta.name else ""
            if casos_hist.indexar_caso(caso_id, json_b, json_c, timestamp):
                ok(f"Caso indexado en vector store: {caso_id}")

    # Mostrar narrativo del dictamen si existe
    if json_c and json_c.get("dictamen", {}).get("analisis_narrativo"):
        titulo("NARRATIVO DEL DICTAMEN (Paso C)", Fore.MAGENTA)
        narrativo = json_c["dictamen"]["analisis_narrativo"]
        for linea in textwrap.wrap(narrativo, width=72):
            print(f"  {linea}")
        print()


if __name__ == "__main__":
    main()