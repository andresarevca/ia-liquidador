"""
vector_store.py
===============
Vector Store + Embeddings para el pipeline de liquidación de siniestros.

Usa:
  - Google text-embedding-004  (ya disponible con GEMINI_API_KEY)
  - ChromaDB local persistente (pip install chromadb)

Colecciones:
  - corpus_legal     : artículos de leyes y ordenanzas chunkeados por artículo
  - casos_historicos : dictámenes pasados para búsqueda de precedentes

USO DESDE LÍNEA DE COMANDOS:
  # Indexar Ley 5016 y la ordenanza integrada (se hace automáticamente en --rag)
  python vector_store.py --init

  # Indexar un PDF de ordenanza municipal
  python vector_store.py --indexar-pdf carpeta/Ordenanza\\ TRANSITO\\ 297-2022.pdf \\
                         --municipio "Katueté"

  # Buscar artículos relevantes
  python vector_store.py --buscar "señal de PARE responsabilidad"

  # Listar documentos indexados
  python vector_store.py --listar

  # Indexar casos históricos desde resultados/
  python vector_store.py --indexar-historicos
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Importaciones opcionales
# ---------------------------------------------------------------------------
try:
    import chromadb
    _CHROMA_OK = True
except ImportError:
    _CHROMA_OK = False

try:
    import google.generativeai as genai
    _GENAI_OK = True
except ImportError:
    _GENAI_OK = False

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
VECTOR_DB_PATH   = "vector_db"           # carpeta local del store persistente
COL_LEGAL        = "corpus_legal"
COL_CASOS        = "casos_historicos"
EMBEDDING_MODEL  = "models/text-embedding-004"
TOP_K_LEGAL      = 6                     # artículos a recuperar por consulta
TOP_K_CASOS      = 3                     # casos similares a recuperar
CHUNK_SIZE       = 900                   # caracteres por chunk (fallback)
CHUNK_OVERLAP    = 120


# ---------------------------------------------------------------------------
# Función de embedding compatible con ChromaDB
# ---------------------------------------------------------------------------

def _embed_documentos(textos: list[str]) -> list[list[float]]:
    """
    Genera embeddings para una lista de textos (indexación).
    Usa task_type RETRIEVAL_DOCUMENT.
    Procesa en lotes de 100 para respetar límites de la API.
    """
    todos = []
    BATCH = 100
    for i in range(0, len(textos), BATCH):
        lote = textos[i : i + BATCH]
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=lote,
            task_type="RETRIEVAL_DOCUMENT",
        )
        embs = result["embedding"]
        # Si el lote tiene un solo elemento, la API devuelve vector plano
        if lote and isinstance(embs[0], float):
            embs = [embs]
        todos.extend(embs)
    return todos


def _embed_consulta(texto: str) -> list[float]:
    """
    Genera embedding para una consulta de búsqueda.
    Usa task_type RETRIEVAL_QUERY (distinto al de indexación).
    """
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=texto,
        task_type="RETRIEVAL_QUERY",
    )
    return result["embedding"]


# ---------------------------------------------------------------------------
# ChromaDB EmbeddingFunction wrapper
# ---------------------------------------------------------------------------

try:
    from chromadb import EmbeddingFunction, Documents, Embeddings

    class _GoogleEF(EmbeddingFunction):
        """Adaptador de Google text-embedding-004 para ChromaDB."""
        def __call__(self, input: Documents) -> Embeddings:          # noqa: A002
            return _embed_documentos(list(input))

except (ImportError, Exception):
    _GoogleEF = None  # type: ignore


def _get_ef():
    if _GoogleEF is None:
        raise ImportError("ChromaDB no instalado. Ejecutar: pip install chromadb")
    return _GoogleEF()


# ---------------------------------------------------------------------------
# Chunking de textos legales
# ---------------------------------------------------------------------------

_ART_RE = re.compile(
    r'(Art[íi]culo?\.?\s*\d+[\w\s\-–—]*?(?=\n|$))',
    re.IGNORECASE | re.MULTILINE,
)


def _chunkear_por_articulos(texto: str, titulo_doc: str) -> list[dict]:
    """
    Divide el texto por artículos (patrón 'Art. N°' o 'Artículo N°').
    Fallback a chunking por tamaño si no hay artículos detectados.
    """
    partes = _ART_RE.split(texto)
    chunks = []

    # Cabecera del documento (antes del primer artículo)
    cabecera = partes[0].strip()
    if cabecera and len(cabecera) > 40:
        chunks.append({
            "texto": cabecera,
            "encabezado": titulo_doc,
            "tipo_chunk": "cabecera",
        })

    # Artículos
    i = 1
    while i < len(partes):
        encabezado = partes[i].strip()
        cuerpo = partes[i + 1].strip() if i + 1 < len(partes) else ""
        contenido = f"{encabezado}\n{cuerpo}".strip()
        if contenido and len(contenido) > 20:
            chunks.append({
                "texto": contenido,
                "encabezado": encabezado,
                "tipo_chunk": "articulo",
            })
        i += 2

    if len(chunks) <= 1:
        # Sin artículos detectados: chunking por tamaño
        return _chunkear_por_tamano(texto, titulo_doc)

    return chunks


def _chunkear_por_tamano(texto: str, titulo_doc: str) -> list[dict]:
    """Chunking de respaldo: tamaño fijo con solapamiento."""
    chunks = []
    inicio = 0
    idx = 0
    while inicio < len(texto):
        segmento = texto[inicio : inicio + CHUNK_SIZE]
        if segmento.strip():
            chunks.append({
                "texto": segmento,
                "encabezado": f"{titulo_doc} [segmento {idx}]",
                "tipo_chunk": "chunk",
            })
        inicio += CHUNK_SIZE - CHUNK_OVERLAP
        idx += 1
    return chunks


# ---------------------------------------------------------------------------
# CorpusLegal
# ---------------------------------------------------------------------------

class CorpusLegal:
    """
    Gestiona el corpus de textos legales (leyes y ordenanzas) en ChromaDB.

    Ejemplo:
        corpus = CorpusLegal()
        corpus.indexar_texto("Ley 5016/14", texto_ley, municipio="Nacional")
        corpus.indexar_pdf(Path("ordenanza.pdf"), municipio="San Lorenzo")
        resultados = corpus.buscar_relevantes("señal de PARE infracción", municipio="San Lorenzo")
    """

    def __init__(self, db_path: str = VECTOR_DB_PATH):
        if not _CHROMA_OK:
            raise ImportError("ChromaDB no instalado. Ejecutar: pip install chromadb")
        self._client = chromadb.PersistentClient(path=db_path)
        self._col = self._client.get_or_create_collection(
            name=COL_LEGAL,
            embedding_function=_get_ef(),
            metadata={"hnsw:space": "cosine"},
        )

    def indexar_texto(
        self,
        titulo: str,
        texto: str,
        municipio: str = "Nacional",
        tipo: str = "LEY",
    ) -> int:
        """
        Indexa un texto legal chunkeado por artículos.
        Omite chunks ya existentes (idempotente).
        Retorna el número de chunks efectivamente indexados.
        """
        chunks = _chunkear_por_articulos(texto, titulo)
        if not chunks:
            return 0

        ids, documentos, metadatas = [], [], []
        for i, chunk in enumerate(chunks):
            doc_id = f"{re.sub(r'[^a-z0-9]', '_', titulo.lower())}_{i}"
            if self._col.get(ids=[doc_id])["ids"]:
                continue  # ya existe
            ids.append(doc_id)
            documentos.append(chunk["texto"])
            metadatas.append({
                "fuente": titulo,
                "municipio": municipio,
                "tipo_doc": tipo,
                "encabezado": chunk.get("encabezado", "")[:200],
                "tipo_chunk": chunk.get("tipo_chunk", "chunk"),
            })

        if ids:
            embeddings = _embed_documentos(documentos)
            self._col.add(
                ids=ids,
                documents=documentos,
                metadatas=metadatas,
                embeddings=embeddings,
            )
        return len(ids)

    def indexar_pdf(self, archivo: Path, municipio: str) -> int:
        """
        Extrae texto de un PDF y lo indexa en el corpus legal.
        Útil para cargar ordenanzas municipales directamente desde archivo.
        """
        texto = _extraer_texto_pdf(archivo)
        if not texto:
            raise ValueError(f"No se pudo extraer texto de: {archivo.name}")
        titulo = archivo.stem
        return self.indexar_texto(titulo, texto, municipio=municipio, tipo="ORDENANZA")

    def buscar_relevantes(
        self,
        consulta: str,
        n: int = TOP_K_LEGAL,
        municipio: Optional[str] = None,
    ) -> list[dict]:
        """
        Búsqueda semántica en el corpus legal.
        Si municipio está dado, filtra por ese municipio Y normas nacionales.
        """
        total = self._col.count()
        if total == 0:
            return []

        where = None
        if municipio and municipio not in ("Nacional", "", None):
            where = {"$or": [
                {"municipio": {"$eq": municipio}},
                {"municipio": {"$eq": "Nacional"}},
            ]}

        emb_q = _embed_consulta(consulta)
        res = self._col.query(
            query_embeddings=[emb_q],
            n_results=min(n, total),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        salida = []
        for doc, meta, dist in zip(
            res["documents"][0],
            res["metadatas"][0],
            res["distances"][0],
        ):
            salida.append({
                "texto": doc,
                "fuente": meta.get("fuente", ""),
                "municipio": meta.get("municipio", ""),
                "encabezado": meta.get("encabezado", ""),
                "similitud": round(1.0 - dist, 4),
            })
        return salida

    def contar(self) -> int:
        return self._col.count()

    def listar_fuentes(self) -> list[str]:
        """Retorna la lista de documentos legales indexados (sin duplicados)."""
        todos = self._col.get(include=["metadatas"])
        fuentes = {m.get("fuente", "") for m in todos["metadatas"]}
        return sorted(fuentes)


# ---------------------------------------------------------------------------
# CasosHistoricos
# ---------------------------------------------------------------------------

class CasosHistoricos:
    """
    Almacena resultados de casos procesados para búsqueda de precedentes.

    Indexa el narrativo del dictamen + descripción dinámica del siniestro,
    junto con metadatos clave del Paso B y Paso C.
    """

    def __init__(self, db_path: str = VECTOR_DB_PATH):
        if not _CHROMA_OK:
            raise ImportError("ChromaDB no instalado. Ejecutar: pip install chromadb")
        self._client = chromadb.PersistentClient(path=db_path)
        self._col = self._client.get_or_create_collection(
            name=COL_CASOS,
            embedding_function=_get_ef(),
            metadata={"hnsw:space": "cosine"},
        )

    def indexar_caso(
        self,
        caso_id: str,
        json_b: dict,
        json_c: Optional[dict],
        timestamp: str = "",
    ) -> bool:
        """
        Indexa un caso por su descripción narrativa y datos clave.
        Retorna True si fue indexado, False si ya existía.
        """
        doc_id = f"{caso_id}_{timestamp}" if timestamp else caso_id
        if self._col.get(ids=[doc_id])["ids"]:
            return False

        siniestro = json_b.get("siniestro", {})
        dictamen  = (json_c or {}).get("dictamen", {})
        poliza    = json_b.get("poliza", {})

        # Texto representativo del caso
        partes_texto = [
            siniestro.get("descripcion_dinamica", ""),
            dictamen.get("analisis_narrativo", ""),
        ]
        descripcion = " ".join(p for p in partes_texto if p).strip()
        if not descripcion:
            descripcion = (
                f"Caso {caso_id}: siniestro en {siniestro.get('municipio', '')} "
                f"cobertura {poliza.get('cobertura_tipo', '')} "
                f"responsabilidad {dictamen.get('responsabilidad_sugerida', '')}"
            )

        metadata = {
            "caso_id":         caso_id,
            "municipio":       (siniestro.get("municipio") or "")[:100],
            "responsabilidad": (dictamen.get("responsabilidad_sugerida") or "")[:60],
            "cobertura_tipo":  (poliza.get("cobertura_tipo") or "")[:40],
            "confianza":       str(dictamen.get("confianza_dictamen") or ""),
            "timestamp":       timestamp,
        }

        emb = _embed_documentos([descripcion[:2000]])
        self._col.add(
            ids=[doc_id],
            documents=[descripcion[:2000]],
            metadatas=[metadata],
            embeddings=emb,
        )
        return True

    def buscar_similares(
        self,
        descripcion_caso: str,
        n: int = TOP_K_CASOS,
    ) -> list[dict]:
        """Encuentra casos históricos similares al caso actual."""
        total = self._col.count()
        if total == 0:
            return []

        emb_q = _embed_consulta(descripcion_caso)
        res = self._col.query(
            query_embeddings=[emb_q],
            n_results=min(n, total),
            include=["documents", "metadatas", "distances"],
        )

        salida = []
        for doc, meta, dist in zip(
            res["documents"][0],
            res["metadatas"][0],
            res["distances"][0],
        ):
            salida.append({
                "caso_id":        meta.get("caso_id", ""),
                "municipio":      meta.get("municipio", ""),
                "responsabilidad":meta.get("responsabilidad", ""),
                "cobertura_tipo": meta.get("cobertura_tipo", ""),
                "similitud":      round(1.0 - dist, 4),
                "resumen":        doc[:300],
            })
        return salida

    def cargar_desde_resultados(
        self, carpeta_resultados: Path = Path("resultados")
    ) -> int:
        """
        Carga todos los casos históricos desde resultados/.
        Útil para inicializar la base con casos ya procesados.
        Retorna el número de casos nuevos indexados.
        """
        indexados = 0
        for sub in sorted(carpeta_resultados.iterdir()):
            if not sub.is_dir():
                continue
            arch_b = sub / "paso_b_extraccion.json"
            arch_c = sub / "paso_c_dictamen.json"
            if not arch_b.exists():
                continue
            try:
                jb = json.loads(arch_b.read_text(encoding="utf-8"))
                jc = json.loads(arch_c.read_text(encoding="utf-8")) if arch_c.exists() else {}
                # caso_id y timestamp desde el nombre de carpeta
                partes = sub.name.rsplit("_", 2)
                caso_id  = partes[0]
                ts       = "_".join(partes[1:]) if len(partes) > 1 else ""
                if self.indexar_caso(caso_id, jb, jc, ts):
                    indexados += 1
            except Exception:
                continue
        return indexados

    def contar(self) -> int:
        return self._col.count()


# ---------------------------------------------------------------------------
# Función de conveniencia para integración con Paso C (RAG)
# ---------------------------------------------------------------------------

def construir_contexto_legal_rag(
    descripcion_caso: str,
    municipio: str,
    corpus: CorpusLegal,
    n: int = TOP_K_LEGAL,
) -> str:
    """
    Recupera los artículos legales más relevantes para el caso e
    construye un bloque de texto listo para insertar en el prompt del Paso C.
    """
    resultados = corpus.buscar_relevantes(descripcion_caso, n=n, municipio=municipio)
    if not resultados:
        return "(Sin artículos recuperados del corpus legal)"

    bloques = []
    for r in resultados:
        fuente   = r["fuente"]
        encabez  = r["encabezado"]
        sim      = r["similitud"]
        texto    = r["texto"]
        bloques.append(f"[{fuente}  ·  sim={sim:.2f}]\n{texto}")

    return "\n\n".join(bloques)


# ---------------------------------------------------------------------------
# Utilidades internas
# ---------------------------------------------------------------------------

def _extraer_texto_pdf(archivo: Path) -> str:
    """Extrae texto de un PDF usando pdfplumber (preferido) o PyPDF2."""
    try:
        import pdfplumber
        with pdfplumber.open(archivo) as pdf:
            return "\n".join(p.extract_text() or "" for p in pdf.pages).strip()
    except ImportError:
        pass
    except Exception:
        pass

    try:
        import PyPDF2
        with open(archivo, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            return "\n".join(
                (p.extract_text() or "") for p in reader.pages
            ).strip()
    except ImportError:
        pass
    except Exception:
        pass

    return ""


# ---------------------------------------------------------------------------
# CLI para administración del corpus
# ---------------------------------------------------------------------------

def _cli_main():
    import os
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[ERROR] GEMINI_API_KEY no configurada.")
        sys.exit(1)
    genai.configure(api_key=api_key)

    parser = argparse.ArgumentParser(description="Gestión del Vector Store legal")
    parser.add_argument("--init",               action="store_true",
                        help="Indexa la Ley 5016/14 y la ordenanza integrada de San Lorenzo")
    parser.add_argument("--indexar-pdf",        metavar="ARCHIVO",
                        help="Ruta a un PDF de ordenanza/ley para indexar")
    parser.add_argument("--municipio",          default="Nacional",
                        help="Municipio al que pertenece el documento (default: Nacional)")
    parser.add_argument("--buscar",             metavar="TEXTO",
                        help="Búsqueda semántica en el corpus legal")
    parser.add_argument("--listar",             action="store_true",
                        help="Lista los documentos legales indexados")
    parser.add_argument("--indexar-historicos", action="store_true",
                        help="Indexa casos históricos desde resultados/")
    parser.add_argument("--db",                 default=VECTOR_DB_PATH,
                        help=f"Ruta de la base de datos (default: {VECTOR_DB_PATH})")
    args = parser.parse_args()

    corpus  = CorpusLegal(db_path=args.db)
    casos   = CasosHistoricos(db_path=args.db)

    if args.init:
        # Importar las constantes del script principal
        try:
            from validar_pipeline_ia import LEY_5016_FRAGMENTOS, ORDENANZA_SAN_LORENZO
            n = corpus.indexar_texto("Ley 5016-14", LEY_5016_FRAGMENTOS, municipio="Nacional", tipo="LEY")
            print(f"[OK] Ley 5016/14 — {n} chunk(s) indexado(s)")
            n = corpus.indexar_texto("Ordenanza 45-2019 San Lorenzo", ORDENANZA_SAN_LORENZO,
                                     municipio="San Lorenzo", tipo="ORDENANZA")
            print(f"[OK] Ordenanza San Lorenzo — {n} chunk(s) indexado(s)")
        except ImportError as e:
            print(f"[ERROR] {e}")
            sys.exit(1)

    if args.indexar_pdf:
        archivo = Path(args.indexar_pdf)
        if not archivo.exists():
            print(f"[ERROR] Archivo no encontrado: {args.indexar_pdf}")
            sys.exit(1)
        n = corpus.indexar_pdf(archivo, municipio=args.municipio)
        print(f"[OK] {archivo.name} → {n} chunk(s) indexado(s) (municipio: {args.municipio})")

    if args.buscar:
        resultados = corpus.buscar_relevantes(args.buscar, n=5)
        print(f"\nResultados para: '{args.buscar}'\n{'─'*60}")
        for r in resultados:
            print(f"\n[{r['fuente']}  ·  sim={r['similitud']:.3f}]")
            print(r["texto"][:400])

    if args.listar:
        fuentes = corpus.listar_fuentes()
        print(f"\nDocumentos en corpus_legal ({corpus.contar()} chunks):")
        for f in fuentes:
            print(f"  • {f}")
        print(f"\nCasos históricos: {casos.contar()}")

    if args.indexar_historicos:
        n = casos.cargar_desde_resultados()
        print(f"[OK] {n} caso(s) histórico(s) indexado(s). Total: {casos.contar()}")


if __name__ == "__main__":
    _cli_main()
