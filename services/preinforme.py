"""
services/preinforme.py
======================
Transforma los resultados del pipeline IA (Paso B + Paso C) en un diccionario
estructurado con todos los campos necesarios para generar el pre-informe de
liquidación de siniestros.

No realiza llamadas adicionales a Gemini; usa los datos ya extraídos.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any


# ---------------------------------------------------------------------------
# Helpers de transformación
# ---------------------------------------------------------------------------

def _fmt_usd(valor) -> str | None:
    if valor is None:
        return None
    try:
        return f"USD. {float(valor):,.2f}"
    except (ValueError, TypeError):
        return str(valor)


def _fmt_gs(valor) -> str | None:
    if valor is None:
        return None
    try:
        return f"Gs. {int(float(valor)):,}".replace(",", ".")
    except (ValueError, TypeError):
        return str(valor)


def _formatear_lugar(siniestro: dict) -> str:
    partes = []
    if siniestro.get("direccion_aproximada"):
        partes.append(siniestro["direccion_aproximada"])
    if siniestro.get("municipio"):
        partes.append(siniestro["municipio"])
    return ", ".join(partes) if partes else ""


def _inferir_detalle_siniestro(siniestro: dict, paso_b: dict) -> str:
    """Intenta construir el detalle del siniestro a partir de los datos disponibles."""
    desc = siniestro.get("descripcion_dinamica", "")
    if desc:
        # Tomar la primera oración como detalle corto
        primera = desc.split(".")[0].strip()
        if primera:
            return primera.upper()
    tipo_via = siniestro.get("tipo_via", "")
    return f"COLISIÓN EN {tipo_via}" if tipo_via else "COLISIÓN - DAÑOS MATERIALES"


def _mapear_documentacion(documentacion: dict, documentos_lista: list) -> list[dict]:
    """Genera la lista de documentación complementaria con estado ENTREGADO/NO ENTREGADO."""
    mapa = [
        ("Presupuesto de reparación de los vehículos afectados", None),
        ("Reclamo formal de los terceros afectados", None),
        ("Acta de procedimiento policial y resultado de alchotest", documentacion.get("denuncia_policial_presente")),
        ("Copia de la carpeta fiscal actualizada", None),
        ("Copia de la cédula verde y habilitación del vehículo asegurado", documentacion.get("cedula_verde_presente")),
        ("Copia del registro y cédula de identidad del conductor asegurado", documentacion.get("licencia_conductor_presente")),
        ("Informe del sistema de rastreo satelital (GPS)", None),
        ("Certificado y acta de defunción de las personas fallecidas", None),
        ("Informe pericial accidentológico", documentacion.get("pericia_tecnica_presente")),
        ("Fotos del evento/daños", documentacion.get("fotos_evento_cantidad", 0) > 0 if documentacion.get("fotos_evento_cantidad") else None),
    ]
    # Si se recibió lista de documentos del backend, marcar los presentes
    nombres_entregados = {d.lower() for d in documentos_lista}
    resultado = []
    for nombre, presente in mapa:
        if presente is True:
            estado = "ENTREGADO"
        elif presente is False:
            estado = "NO ENTREGADO"
        elif any(k in nombre.lower() for k in nombres_entregados):
            estado = "ENTREGADO"
        else:
            estado = "NO ENTREGADO"
        resultado.append({"nombre": nombre, "estado": estado})
    return resultado


def _extraer_normas(infracciones: list) -> list[str]:
    """Extrae artículos legales únicos citados en las infracciones."""
    normas = []
    vistos = set()
    for inf in infracciones:
        for campo in ("articulo_ley_5016", "articulo_ordenanza"):
            val = inf.get(campo)
            if val and val not in vistos:
                vistos.add(val)
                normas.append(val)
    return normas


def _mapear_coberturas(poliza: dict, dictamen: dict) -> list[dict]:
    """Construye la lista de coberturas afectadas en el formato del pre-informe."""
    coberturas = []
    cobertura_tipo = poliza.get("cobertura_tipo", "")
    suma = poliza.get("suma_asegurada")
    franquicia = poliza.get("franquicia")

    # Cap A — Responsabilidad Civil (siempre presente si hay RC)
    if cobertura_tipo in ("RESPONSABILIDAD_CIVIL", "TODO_RIESGO", "PARCIAL"):
        coberturas.append({
            "capitulo": "A",
            "titulo": "RESPONSABILIDAD CIVIL HACIA TERCEROS",
            "items": [
                "Lesiones o Muerte a personas",
                "Daños Materiales causados a terceros no transportados",
            ],
        })

    # Cap B — Daños Materiales
    if cobertura_tipo in ("TODO_RIESGO", "PARCIAL"):
        coberturas.append({
            "capitulo": "B",
            "titulo": "DAÑOS MATERIALES AL VEHÍCULO",
            "monto_usd": _fmt_usd(suma),
            "items": [
                f"Daños Materiales sufridos por el vehículo asegurado hasta {_fmt_usd(suma) or 'el valor del vehículo al momento del siniestro'}",
            ],
        })

    # Accidentes personales
    coberturas.append({
        "capitulo": "ACCIDENTES PERSONALES",
        "titulo": "ACCIDENTES PERSONALES DE LOS OCUPANTES",
        "items": ["Gastos Médicos por cada ocupante"],
    })

    return coberturas


def _formatear_responsabilidad_conclusion(responsabilidad: str, dictamen: dict) -> str:
    """Formatea el párrafo de conclusión sobre responsabilidad."""
    mapeo = {
        "TERCERO_RESPONSABLE": "La máxima responsabilidad recae sobre el conductor del vehículo tercero.",
        "ASEGURADO_RESPONSABLE": "La máxima responsabilidad recae sobre el conductor del vehículo asegurado.",
        "RESPONSABILIDAD_COMPARTIDA": "Se determina responsabilidad concurrente entre ambos conductores.",
        "CASO_FORTUITO": "El siniestro se determina como caso fortuito.",
        "INDETERMINADO": "La responsabilidad no pudo determinarse con los elementos disponibles.",
    }
    base = mapeo.get(responsabilidad, "")
    pct_a = dictamen.get("porcentaje_responsabilidad_asegurado")
    pct_t = dictamen.get("porcentaje_responsabilidad_tercero")
    if pct_a is not None and pct_t is not None:
        base += f" Asegurado: {pct_a}% / Tercero: {pct_t}%."
    return base


# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------

def generar_datos_preinforme(
    caso_id: str,
    paso_b: dict,
    paso_c: dict | None,
    metadatos: dict | None = None,
) -> dict[str, Any]:
    """
    Transforma paso_b + paso_c en un dict con todos los campos del pre-informe.

    metadatos puede contener:
      - nro_siniestro: str
      - solicitante: str        (si no viene en la póliza)
      - asegurado: str          (nombre del asegurado)
      - fecha_denuncia: str
      - fecha_designacion: str
      - actuaciones: list[dict] ({"fecha": str, "descripcion": str})
      - documentos: list[str]   (nombres de archivos entregados)
    """
    metadatos = metadatos or {}
    paso_c = paso_c or {}

    siniestro = paso_b.get("siniestro", {})
    vehiculos = paso_b.get("vehiculos", [])
    poliza = paso_b.get("poliza", {})
    documentacion = paso_b.get("documentacion", {})
    monto_danios = paso_b.get("monto_danios", {})
    conflictos = paso_b.get("conflictos", [])

    dictamen = paso_c.get("dictamen", {}) if paso_c else {}

    # Vehículos por rol
    v_asegurado = next((v for v in vehiculos if v.get("rol") == "ASEGURADO"), {})
    v_terceros = [v for v in vehiculos if v.get("rol") != "ASEGURADO"]
    v_tercero = v_terceros[0] if v_terceros else {}

    responsabilidad = dictamen.get("responsabilidad_sugerida", "INDETERMINADO")
    cobertura_aplica = dictamen.get("cobertura_aplica", "condicional")
    monto_liquidar = dictamen.get("monto_sugerido_liquidar")
    monto_pericia = monto_danios.get("estimacion_pericia")

    # Indemnización asegurado
    corresponde_asegurado = cobertura_aplica is True or cobertura_aplica == "true" or cobertura_aplica == "condicional"
    corresponde_tercero = responsabilidad in ("TERCERO_RESPONSABLE",)

    # Coberturas afectadas
    coberturas = _mapear_coberturas(poliza, dictamen)

    return {
        # Identificadores
        "caso_id": caso_id,
        "nro_siniestro": metadatos.get("nro_siniestro", caso_id),
        "generado_en": datetime.now().strftime("%d/%m/%Y %H:%M"),

        # Cabecera del informe
        "solicitante": poliza.get("aseguradora") or metadatos.get("solicitante", ""),
        "asegurado": metadatos.get("asegurado") or v_asegurado.get("conductor_nombre", ""),
        "nro_poliza": poliza.get("numero", ""),
        "seccion": "AUTOMOVILES",
        "periodo_desde": poliza.get("vigencia_desde", ""),
        "periodo_hasta": poliza.get("vigencia_hasta", ""),

        # Siniestro
        "detalle_siniestro": _inferir_detalle_siniestro(siniestro, paso_b),
        "lugar_siniestro": _formatear_lugar(siniestro),
        "fecha_siniestro": siniestro.get("fecha_hora", ""),
        "condicion_climatica": siniestro.get("condicion_climatica", ""),

        # Fechas del proceso
        "fecha_denuncia": metadatos.get("fecha_denuncia", siniestro.get("fecha_hora", "")),
        "fecha_designacion": metadatos.get("fecha_designacion", ""),
        "ultima_documentacion": metadatos.get("ultima_documentacion", ""),

        # Vehículo asegurado
        "vehiculo_asegurado": {
            "marca": v_asegurado.get("marca", ""),
            "modelo": v_asegurado.get("modelo", ""),
            "año": v_asegurado.get("año", ""),
            "color": v_asegurado.get("color", ""),
            "matricula": v_asegurado.get("matricula", ""),
            "vin": "",  # no viene en paso_b estándar
        },
        "conductor_asegurado": {
            "nombre": v_asegurado.get("conductor_nombre", ""),
            "ci": v_asegurado.get("conductor_ci", ""),
            "licencia": v_asegurado.get("licencia_numero", ""),
            "categoria": v_asegurado.get("licencia_categoria", ""),
        },

        # Vehículo tercero
        "vehiculo_tercero": {
            "marca": v_tercero.get("marca", ""),
            "modelo": v_tercero.get("modelo", ""),
            "año": v_tercero.get("año", ""),
            "color": v_tercero.get("color", ""),
            "matricula": v_tercero.get("matricula", ""),
        } if v_tercero else None,
        "conductor_tercero": {
            "nombre": v_tercero.get("conductor_nombre", ""),
            "ci": v_tercero.get("conductor_ci", ""),
        } if v_tercero else None,

        # Indemnización al asegurado
        "corresponde_indemnizar_asegurado": corresponde_asegurado,
        "estimacion_danios_materiales": _fmt_usd(monto_liquidar or monto_pericia),
        "estimacion_gastos_medicos": _fmt_usd(metadatos.get("gastos_medicos_usd")),
        "razon_cobertura_asegurado": dictamen.get("razon_cobertura", ""),

        # Indemnización al tercero
        "corresponde_indemnizar_tercero": corresponde_tercero,
        "exposicion_maxima_rc": _fmt_usd(metadatos.get("exposicion_maxima_rc_usd")),
        "razon_no_cobertura_tercero": "" if corresponde_tercero else "El tercero incurrió en culpa grave al circular a velocidad excesiva.",

        # Documentación complementaria
        "documentacion_complementaria": _mapear_documentacion(
            documentacion,
            metadatos.get("documentos", []),
        ),

        # Actuaciones
        "actuaciones": metadatos.get("actuaciones", []),

        # Coberturas de la póliza
        "coberturas_afectadas": coberturas,
        "poliza_cobertura_tipo": poliza.get("cobertura_tipo", ""),
        "poliza_suma_asegurada": _fmt_usd(poliza.get("suma_asegurada")),
        "franquicia": "No posee." if not poliza.get("franquicia") else _fmt_usd(poliza.get("franquicia")),
        "exposicion_maxima_total": _fmt_usd(poliza.get("suma_asegurada")),

        # Narrativo del siniestro (del Paso B)
        "dinamica_siniestro": siniestro.get("descripcion_dinamica", ""),

        # Conclusión pericial (del Paso C — analisis_narrativo)
        "conclusion_pericial": dictamen.get("analisis_narrativo", ""),

        # Análisis de cobertura
        "responsabilidad_conclusion": _formatear_responsabilidad_conclusion(responsabilidad, dictamen),
        "cobertura_aplica": cobertura_aplica,
        "franquicia_aplica": dictamen.get("franquicia_aplica", False),
        "monto_sugerido_liquidar": _fmt_usd(monto_liquidar),

        # Infracciones y normas legales
        "infracciones_detectadas": dictamen.get("infracciones_detectadas", []),
        "normas_legales_afectadas": _extraer_normas(dictamen.get("infracciones_detectadas", [])),

        # Resultado final
        "responsabilidad_sugerida": responsabilidad,
        "porcentaje_asegurado": dictamen.get("porcentaje_responsabilidad_asegurado"),
        "porcentaje_tercero": dictamen.get("porcentaje_responsabilidad_tercero"),
        "alertas_liquidador": dictamen.get("alertas_liquidador", []),
        "confianza_dictamen": dictamen.get("confianza_dictamen", 0),
        "calidad_extraccion": paso_b.get("calidad_extraccion", {}),
        "conflictos": conflictos,
    }
