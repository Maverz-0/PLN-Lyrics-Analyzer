# -*- coding: utf-8 -*-
"""
Analizadores de letras (multi-idioma) para PLN-Lyrics-Analyzer.

Incluye:
- Detección de idioma (langdetect)
- Gestión de modelos spaCy por idioma (con cache y fallbacks)
- Análisis: porcentaje de adjetivos, riqueza léxica (TTR),
  frecuencia de pronombres, detección de rimas (ignorando paréntesis/ad-libs),
  repetición de versos, palabras más frecuentes,
  detección de temas (zero-shot + fallback), detección de metáforas (heurístico),
  NER multi-idioma con spaCy, sentimiento PRO con XLM-RoBERTa (Transformers) robusto.
"""

from __future__ import annotations

# =========================
# Imports estándar
# =========================
import os
import re
import unicodedata
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Any, Dict, List, Tuple

# =========================
# Terceros
# =========================
import spacy

# Detección de idioma
from langdetect import detect_langs, DetectorFactory
DetectorFactory.seed = 0  # resultados reproducibles

# Transformers (opcional; manejamos ausencia)
try:
    from transformers import pipeline  # type: ignore
except Exception:  # pragma: no cover
    pipeline = None  # se usará fallback donde proceda

# =========================
# Config / Constantes
# =========================

# Modelos spaCy por idioma
SPACY_MODELS = {
    "es": "es_core_news_sm",
    "en": "en_core_web_sm",
    "fr": "fr_core_news_sm",
    "pt": "pt_core_news_sm",
    "it": "it_core_news_sm",
    "de": "de_core_news_sm",
}

# Fallback de pronombres por idioma
PRON_FALLBACK = {
    "es": {
        "1": {"yo", "me", "mí", "conmigo", "nosotros", "nosotras", "nos",
              "mío", "mía", "míos", "mías", "nuestro", "nuestra",
              "nuestros", "nuestras"},
        "2": {"tú", "te", "ti", "contigo", "vosotros", "vosotras", "os",
              "usted", "ustedes", "tuyo", "tuya", "tuyos", "tuyas",
              "vuestro", "vuestra", "vuestros", "vuestras", "su", "sus"},
        "3": {"él", "ella", "ello", "se", "sí", "consigo", "ellos", "ellas",
              "les", "lo", "la", "los", "las", "le", "su", "sus"}
    },
    "en": {
        "1": {"i", "me", "my", "mine", "we", "us", "our", "ours"},
        "2": {"you", "your", "yours"},
        "3": {"he", "him", "his", "she", "her", "hers", "it", "its",
              "they", "them", "their", "theirs"}
    },
    "fr": {
        "1": {"je", "me", "moi", "nous", "notre", "nos",
              "mien", "mienne", "miens", "miennes"},
        "2": {"tu", "te", "toi", "vous", "votre", "vos",
              "tien", "tienne", "tiens", "tiennes"},
        "3": {"il", "elle", "on", "se", "lui", "eux", "elles",
              "leur", "leurs", "son", "sa", "ses"}
    },
}

# Vocales por idioma (para rimas)
_VOWELS_BY_LANG = {
    "default": set("aeiou"),
    "es": set("aeiou"),
    "en": set("aeiouy"),
    "fr": set("aeiouy"),
    "pt": set("aeiou"),
    "it": set("aeiou"),
    "de": set("aeiouy"),
}

# Etiquetas NER a nombre humano
_LABEL_HUMANA = {
    "PER": "Persona", "PERSON": "Persona",
    "ORG": "Organización",
    "LOC": "Lugar", "GPE": "Lugar",
    "PRODUCT": "Producto",
    "WORK_OF_ART": "Obra",
    "EVENT": "Evento",
    "LANGUAGE": "Idioma",
    "DATE": "Fecha", "TIME": "Hora",
    "MONEY": "Dinero",
    "NORP": "Gentilicio/Grupo",
    "FAC": "Instalación",
    "LAW": "Ley",
    "CARDINAL": "Número", "QUANTITY": "Cantidad", "PERCENT": "Porcentaje", "ORDINAL": "Ordinal"
}

# =========================
# Utilidades generales
# =========================

def detectar_idioma(texto: str) -> Dict[str, Any]:
    """Devuelve {'codigo': 'es'/'en'/..., 'confianza': float}."""
    try:
        candidatos = detect_langs(texto)
        best = max(candidatos, key=lambda x: x.prob)
        return {"codigo": best.lang, "confianza": round(best.prob, 3)}
    except Exception:
        return {"codigo": "es", "confianza": 0.0}

@lru_cache(maxsize=8)
def get_nlp(lang_code: str):
    """Carga y cachea el modelo de spaCy para un código de idioma."""
    modelo = SPACY_MODELS.get(lang_code)
    if not modelo:
        return None
    try:
        return spacy.load(modelo)
    except Exception:
        return None

def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")

def _normalize_line(line: str) -> str:
    """Limpia líneas: elimina [Sección], normaliza espacios y deja letras/apóstrofos."""
    line = line.strip()
    if not line:
        return ""
    if re.match(r"^\[.*?\]$", line):
        return ""
    line = re.sub(r"[^A-Za-zÁÉÍÓÚÜÑáéíóúüñ'’ ]+", " ", line)
    line = re.sub(r"\s+", " ", line)
    return line.strip()

def _last_word(line: str) -> str:
    parts = line.split()
    return parts[-1] if parts else ""

def _lang_vowels(lang_code: str):
    return _VOWELS_BY_LANG.get(lang_code, _VOWELS_BY_LANG["default"])

def _clean_lines(texto: str) -> List[str]:
    """Filtra líneas no vacías conservando orden."""
    return [l.strip() for l in texto.splitlines() if l.strip()]

def _chunk_text(lines: List[str], max_chars: int = 320) -> List[str]:
    """Agrupa líneas en trozos <= max_chars respetando saltos."""
    chunks: List[str] = []
    cur = ""
    for line in lines:
        if len(cur) + len(line) + 1 > max_chars:
            if cur:
                chunks.append(cur)
                cur = ""
        cur = (cur + "\n" + line) if cur else line
    if cur:
        chunks.append(cur)
    return chunks if chunks else [" ".join(lines)[:max_chars]]

def _weighted_avg(values: List[Tuple[float, int]]) -> float:
    num = sum(v * w for v, w in values)
    den = sum(w for _, w in values) or 1
    return num / den

# =========================
# Sentimiento PRO (Transformers) — robusto
# =========================

# Cache de HF local (útil para docker)
HF_CACHE = os.path.join(os.path.dirname(__file__), ".hf_cache")
os.environ.setdefault("HF_HOME", HF_CACHE)

SENT_MODEL_ID = os.getenv("SENT_MODEL_ID", "cardiffnlp/twitter-xlm-roberta-base-sentiment")
SENT_MODEL_LOCAL = os.getenv("SENT_MODEL_LOCAL", "")  # e.g. "/models/xlm-roberta-sentiment"

# Estado y carga perezosa
_SENT_PIPE = None
_SENT_OK = pipeline is not None

def _get_sent_pipe():
    """Crea el pipeline de sentimiento en diferido (usa carpeta local si se define)."""
    global _SENT_PIPE
    if not _SENT_OK:
        raise RuntimeError("Transformers no disponible en el entorno.")
    if _SENT_PIPE is None:
        model_ref = SENT_MODEL_LOCAL if SENT_MODEL_LOCAL else SENT_MODEL_ID
        _SENT_PIPE = pipeline(
            "sentiment-analysis",
            model=model_ref,
            tokenizer=model_ref,
            device=-1  # CPU; usa 0 si tienes GPU
        )
    return _SENT_PIPE

def _dist_to_scores(dist: List[Dict[str, Any]], pipe=None) -> Tuple[float, float, float, float]:
    """
    Normaliza la distribución del pipeline a (score[-1,1], p_pos, p_neu, p_neg),
    soportando etiquetas 'positive/neutral/negative' o 'LABEL_0/1/2'.
    """
    raw = {str(x["label"]).lower(): float(x["score"]) for x in dist}

    # Caso ideal
    p_pos = raw.get("positive")
    p_neu = raw.get("neutral")
    p_neg = raw.get("negative")

    if p_pos is None or p_neu is None or p_neg is None:
        mapped = {}
        id2label = getattr(getattr(getattr(pipe, "model", None), "config", None), "id2label", None)
        if isinstance(id2label, dict):
            for k, v in id2label.items():
                lab_key = f"label_{k}".lower()
                vlow = str(v).lower()
                if "pos" in vlow:
                    mapped["positive"] = raw.get(lab_key, 0.0)
                elif "neu" in vlow:
                    mapped["neutral"] = raw.get(lab_key, 0.0)
                elif "neg" in vlow:
                    mapped["negative"] = raw.get(lab_key, 0.0)
        if not mapped:
            mapped = {
                "negative": raw.get("label_0", 0.0),
                "neutral":  raw.get("label_1", 0.0),
                "positive": raw.get("label_2", 0.0),
            }
        p_pos = mapped.get("positive", 0.0)
        p_neu = mapped.get("neutral", 0.0)
        p_neg = mapped.get("negative", 0.0)

    s = p_pos - p_neg  # [-1,1]
    return s, p_pos, p_neu, p_neg

def analizar_sentimiento_pro(texto: str) -> Dict[str, Any]:
    """
    Sentimiento [-1,1] con XLM-RoBERTa; promedio por trozos.
    Nunca lanza excepción: si falla, devuelve neutro con 'warning'.
    """
    base = {
        "tipo": "sentimiento",
        "modelo": SENT_MODEL_LOCAL or SENT_MODEL_ID,
        "score": 0.0,
        "resumen": "neutro",
        "distribucion": {"positivo": 0, "neutral": 100, "negativo": 0},
        "segmentos_top": {"positivos": [], "negativos": []},
        "total_segmentos": 0
    }

    if not texto or not texto.strip():
        return base

    if not _SENT_OK:
        return {**base, "warning": "Transformers no disponible en el entorno."}

    try:
        pipe = _get_sent_pipe()
    except Exception as e:
        return {**base, "warning": f"No se pudo cargar el modelo: {e}"}

    try:
        lines = _clean_lines(texto)
        chunks = _chunk_text(lines, max_chars=320)
        outputs = pipe(chunks, truncation=True, return_all_scores=True, batch_size=16)

        weights = [max(1, len(c)) for c in chunks]
        scores_w: List[Tuple[float, int]] = []
        pos_w: List[Tuple[float, int]] = []
        neu_w: List[Tuple[float, int]] = []
        neg_w: List[Tuple[float, int]] = []
        segs: List[Dict[str, Any]] = []

        for ch, dist, w in zip(chunks, outputs, weights):
            s, ppos, pneu, pneg = _dist_to_scores(dist, pipe)
            scores_w.append((s, w))
            pos_w.append((ppos, w))
            neu_w.append((pneu, w))
            neg_w.append((pneg, w))
            segs.append({
                "texto": ch if len(ch) <= 140 else ch[:137] + "…",
                "score": round(s, 3),
                "pos": round(ppos, 3),
                "neu": round(pneu, 3),
                "neg": round(pneg, 3)
            })

        global_score = _weighted_avg(scores_w)
        g_pos = _weighted_avg(pos_w)
        g_neu = _weighted_avg(neu_w)
        g_neg = _weighted_avg(neg_w)

        resumen = "positivo" if global_score > 0.2 else ("negativo" if global_score < -0.2 else "neutro")
        pos_top = sorted([s for s in segs if s["score"] > 0.1], key=lambda x: x["score"], reverse=True)[:2]
        neg_top = sorted([s for s in segs if s["score"] < -0.1], key=lambda x: x["score"])[:2]

        return {
            **base,
            "score": round(global_score, 3),
            "resumen": resumen,
            "distribucion": {
                "positivo": int(round(g_pos * 100)),
                "neutral": int(round(g_neu * 100)),
                "negativo": int(round(g_neg * 100)),
            },
            "segmentos_top": {"positivos": pos_top, "negativos": neg_top},
            "total_segmentos": len(chunks)
        }
    except Exception as e:
        return {**base, "warning": f"Error en inferencia: {e}"}

# =========================
# Análisis: morfosintaxis y léxico
# =========================

def porcentaje_adjetivos(texto: str, lang_code: str | None = None) -> Dict[str, Any]:
    if not lang_code:
        lang_code = detectar_idioma(texto)["codigo"]
    nlp = get_nlp(lang_code) or spacy.load("es_core_news_sm")
    doc = nlp(texto)
    total_palabras = len([t for t in doc if t.is_alpha])
    adjetivos = len([t for t in doc if t.pos_ == "ADJ"])
    if total_palabras == 0:
        return {"tipo": "porcentaje_adjetivos", "porcentaje": 0, "adjetivos": 0, "total": 0, "idioma": lang_code}
    porcentaje = round((adjetivos / total_palabras) * 100, 2)
    return {
        "tipo": "porcentaje_adjetivos",
        "porcentaje": porcentaje,
        "adjetivos": adjetivos,
        "total": total_palabras,
        "idioma": lang_code
    }

def riqueza_lexica(
    texto: str,
    usar_lemmas: bool = True,
    excluir_stopwords: bool = True,
    solo_palabras_contenido: bool = True,
    lang_code: str | None = None
) -> Dict[str, Any]:
    if not lang_code:
        lang_code = detectar_idioma(texto)["codigo"]
    nlp = get_nlp(lang_code) or get_nlp("es") or spacy.load("es_core_news_sm")
    doc = nlp(texto)

    stopwords = nlp.Defaults.stop_words if excluir_stopwords else set()
    content_pos = {"NOUN", "VERB", "ADJ", "ADV"} if solo_palabras_contenido else None

    def es_valido(t) -> bool:
        if not t.is_alpha:
            return False
        if excluir_stopwords and t.text.lower() in stopwords:
            return False
        if content_pos and t.pos_ not in content_pos:
            return False
        return True

    tokens_validos = [t for t in doc if es_valido(t)]
    total = len(tokens_validos)
    if total == 0:
        return {
            "tipo": "riqueza_lexica",
            "metodo": "TTR",
            "ttr": 0.0,
            "unicos": 0,
            "total": 0,
            "idioma": lang_code,
            "opciones": {
                "usar_lemmas": usar_lemmas,
                "excluir_stopwords": excluir_stopwords,
                "solo_palabras_contenido": solo_palabras_contenido
            }
        }

    tipos = {(t.lemma_ if usar_lemmas else t.text).lower() for t in tokens_validos}
    unicos = len(tipos)
    ttr = round(unicos / total, 3)
    return {
        "tipo": "riqueza_lexica",
        "metodo": "TTR",
        "ttr": ttr,
        "unicos": unicos,
        "total": total,
        "idioma": lang_code,
        "opciones": {
            "usar_lemmas": usar_lemmas,
            "excluir_stopwords": excluir_stopwords,
            "solo_palabras_contenido": solo_palabras_contenido
        }
    }

def frecuencia_pronombres(texto: str, incluir_posesivos: bool = True, lang_code: str | None = None) -> Dict[str, Any]:
    """
    Preferencia: POS/Morph con spaCy; fallback: diccionario léxico por idioma.
    """
    if not lang_code:
        lang_code = detectar_idioma(texto)["codigo"]
    nlp = get_nlp(lang_code)

    contadores = {"1": 0, "2": 0, "3": 0, "otros": 0}
    ejemplos = {"1": [], "2": [], "3": [], "otros": []}

    if nlp is None:
        # Fallback léxico
        lex = PRON_FALLBACK.get(lang_code, {})
        tokens = [tok.lower() for tok in texto.split()]
        total = 0
        for p in ("1", "2", "3"):
            s = lex.get(p, set())
            c = sum(1 for tok in tokens if tok.strip(".,;:!?¡¿()[]\"'") in s)
            contadores[p] = c
            total += c
            for tok in tokens:
                w = tok.strip(".,;:!?¡¿()[]\"'").lower()
                if w in s and len(ejemplos[p]) < 8:
                    ejemplos[p].append(w)
        contadores["otros"] = 0
        total = sum(contadores.values())
        porcentajes = {k: (0.0 if total == 0 else round(contadores[k] * 100 / total, 2)) for k in contadores.keys()}
        return {
            "tipo": "frecuencia_pronombres",
            "total": total,
            "conteos": contadores,
            "porcentajes": porcentajes,
            "ejemplos": ejemplos,
            "incluye_posesivos": incluir_posesivos,
            "idioma": lang_code,
            "metodo": "fallback_lexico"
        }

    # Con modelo spaCy
    doc = nlp(texto)

    def es_pronombre_interes(t) -> bool:
        if not t.is_alpha:
            return False
        if t.pos_ == "PRON":
            return True
        if incluir_posesivos and t.pos_ == "DET" and "PronType=Prs" in t.morph:
            return True
        return False

    tokens = [t for t in doc if es_pronombre_interes(t)]
    total = len(tokens)

    fallback_persona = {
        "mí": "1", "conmigo": "1",
        "ti": "2", "contigo": "2",
        "sí": "3", "consigo": "3",
        "se": "3"
    }

    for t in tokens:
        persona = t.morph.get("Person")
        if persona:
            p = persona[0]
        else:
            p = fallback_persona.get(t.text.lower(), None)

        if p in {"1", "2", "3"}:
            contadores[p] += 1
            if len(ejemplos[p]) < 8:
                ejemplos[p].append(t.text)
        else:
            contadores["otros"] += 1
            if len(ejemplos["otros"]) < 8:
                ejemplos["otros"].append(t.text)

    porcentajes = {"1": 0.0, "2": 0.0, "3": 0.0, "otros": 0.0} if total == 0 else \
                  {k: round((v / total) * 100, 2) for k, v in contadores.items()}

    return {
        "tipo": "frecuencia_pronombres",
        "total": total,
        "conteos": contadores,
        "porcentajes": porcentajes,
        "ejemplos": ejemplos,
        "incluye_posesivos": incluir_posesivos,
        "idioma": lang_code,
        "metodo": "spacy_pos_morph"
    }

# =========================
# Análisis: rimas / repetición / frecuencia
# =========================

def _rhyme_key_consonant(word_norm: str, tail_len: int = 3) -> str:
    return word_norm[-tail_len:] if len(word_norm) >= tail_len else word_norm

def _rhyme_key_assonant(word_norm: str, vowels: set[str]) -> str:
    tail = word_norm[-5:]
    return "".join(ch for ch in tail if ch in vowels)

# --- NUEVO: quitar paréntesis/ad-libs solo para rimas ---
_PAREN_BLOCK_RE = re.compile(r"\([^)]*\)")         # ( ... )
_PAREN_TAIL_OPEN_RE = re.compile(r"\([^\)]*$")     # '(' sin cerrar al final

def _strip_parentheticals(line: str) -> str:
    """
    Elimina todo el contenido entre paréntesis de un verso.
    - Borra (ad-libs), (x2), etc.
    - Si hay '(' sin cerrar al final, también se elimina.
    """
    s = line
    # elimina cualquier bloque ( ... )
    while True:
        new = _PAREN_BLOCK_RE.sub(" ", s)
        if new == s:
            break
        s = new
    # elimina '(' colgante al final
    s = _PAREN_TAIL_OPEN_RE.sub("", s)
    # normaliza espacios
    s = re.sub(r"\s+", " ", s).strip()
    return s

def deteccion_rimas(texto: str, lang_code: str | None = None, tail_len: int = 3) -> Dict[str, Any]:
    """
    Rimas por fin de verso: clave consonante y asonante; esquema A/B/C priorizando consonante.
    - **Ignora paréntesis/ad-libs** al final o dentro de los versos (p.ej. "(yeah)", "(x2)").
    """
    if not lang_code:
        lang_code = detectar_idioma(texto)["codigo"]
    _ = get_nlp(lang_code) or get_nlp("es") or spacy.load("es_core_news_sm")  # asegurar recursos (stopwords, etc.)
    vowels = _lang_vowels(lang_code)

    raw_lines = [ln for ln in texto.splitlines()]

    # 1) Quitar paréntesis/ad-libs antes de normalizar
    pre = [_strip_parentheticals(ln) for ln in raw_lines]

    # 2) Normalización (quita etiquetas y puntuación, deja letras/apóstrofos)
    lines = [_normalize_line(ln) for ln in pre]

    # 3) Filtrar vacías y construir índice hacia líneas originales
    idx_map = [i for i, ln in enumerate(lines) if ln]
    lines = [ln for ln in lines if ln]

    if not lines:
        return {
            "tipo": "deteccion_rimas", "idioma": lang_code,
            "num_versos": 0, "esquema": [], "grupos": [], "stats": {}
        }

    # Última palabra normalizada
    last_words = []
    for ln in lines:
        lw = _last_word(ln)
        lw = _strip_accents(lw).lower()
        last_words.append(lw)

    # Claves de rima
    keys_cons = [_rhyme_key_consonant(w, tail_len=tail_len) for w in last_words]
    keys_asso = [_rhyme_key_assonant(w, vowels=vowels) for w in last_words]

    # Agrupar
    group_map: Dict[int, str] = {}
    clusters: List[Dict[str, Any]] = []
    label_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    next_label_idx = 0

    # Grupos consonantes (>=2)
    cons_groups = defaultdict(list)
    for i, k in enumerate(keys_cons):
        cons_groups[k].append(i)
    used = set()
    for k, members in cons_groups.items():
        if len(members) >= 2 and k != "":
            label = label_chars[next_label_idx % len(label_chars)]
            next_label_idx += 1
            for m in members:
                group_map[m] = label
                used.add(m)
            clusters.append({"tipo": "consonante", "clave": k, "miembros": members})

    # Grupos asonantes para no agrupados (>=2)
    asso_groups = defaultdict(list)
    for i, k in enumerate(keys_asso):
        if i in used:
            continue
        asso_groups[k].append(i)
    for k, members in asso_groups.items():
        if len(members) >= 2 and k != "":
            label = label_chars[next_label_idx % len(label_chars)]
            next_label_idx += 1
            for m in members:
                group_map[m] = label
                used.add(m)
            clusters.append({"tipo": "asonante", "clave": k, "miembros": members})

    # Etiquetado de esquema
    esquema = [group_map.get(i, "-") for i in range(len(lines))]

    # Métricas
    versos_con_rima = sum(1 for e in esquema if e != "-")
    stats = {
        "versos": len(lines),
        "versos_con_rima": versos_con_rima,
        "porcentaje_con_rima": round(versos_con_rima * 100 / len(lines), 2),
        "num_grupos": len(clusters),
        "rimas_consonantes": sum(1 for c in clusters if c["tipo"] == "consonante"),
        "rimas_asonantes": sum(1 for c in clusters if c["tipo"] == "asonante"),
    }

    # Ejemplos y mapeo a índices originales
    for cl in clusters:
        cl["indices"] = [idx_map[i] for i in cl["miembros"]]
        cl["ejemplos"] = [raw_lines[idx_map[i]] for i in cl["miembros"][:3]]

    return {
        "tipo": "deteccion_rimas",
        "idioma": lang_code,
        "num_versos": len(lines),
        "esquema": esquema,
        "grupos": clusters,
        "stats": stats,
    }

def repeticion_versos(texto: str, sim_umbral: float = 0.85, max_versos: int = 250) -> Dict[str, Any]:
    """
    Agrupa líneas repetidas o muy similares (posible estribillo).
    Coincidencia exacta + similitud difusa (SequenceMatcher).
    """
    raw_lines = [ln for ln in texto.splitlines()]
    lines = [_normalize_line(ln).lower() for ln in raw_lines]
    idx_map = [i for i, ln in enumerate(lines) if ln]
    lines = [ln for ln in lines if ln][:max_versos]
    idx_map = idx_map[:len(lines)]

    if not lines:
        return {"tipo": "repeticion_versos", "grupos": [], "posible_estribillo": None, "stats": {"versos": 0}}

    # Agrupación exacta
    exact = defaultdict(list)
    for i, ln in enumerate(lines):
        exact[ln].append(i)

    grupos: List[Dict[str, Any]] = []
    usados = set()
    for texto_norm, members in exact.items():
        if len(members) >= 2:
            grupos.append({"tipo": "exacta", "miembros": members, "texto": texto_norm})
            usados.update(members)

    # Similitud difusa entre no usados
    rem = [i for i in range(len(lines)) if i not in usados]
    visited = set()
    for i in rem:
        if i in visited:
            continue
        cluster = [i]
        visited.add(i)
        for j in rem:
            if j in visited:
                continue
            s = __import__("difflib").SequenceMatcher(None, lines[i], lines[j]).ratio()
            if s >= sim_umbral:
                cluster.append(j)
                visited.add(j)
        if len(cluster) >= 2:
            grupos.append({"tipo": "difusa", "miembros": cluster, "texto": lines[i]})

    grupos.sort(key=lambda g: len(g["miembros"]), reverse=True)

    posible_estribillo = None
    if grupos:
        g = grupos[0]
        originales = [idx_map[i] for i in g["miembros"]]
        posibles_offsets = [originales[k + 1] - originales[k] for k in range(len(originales) - 1)]
        periodicidad = Counter(posibles_offsets).most_common(1)[0][0] if posibles_offsets else None
        posible_estribillo = {
            "tam_grupo": len(g["miembros"]),
            "tipo": g["tipo"],
            "primer_linea": raw_lines[originales[0]],
            "apariciones": originales,
            "periodicidad_aprox": periodicidad
        }

    for g in grupos:
        g["indices"] = [idx_map[i] for i in g["miembros"]]
        g["ejemplos"] = [raw_lines[idx_map[i]] for i in g["miembros"][:3]]

    return {
        "tipo": "repeticion_versos",
        "grupos": grupos,
        "posible_estribillo": posible_estribillo,
        "stats": {"versos": len(lines), "num_grupos": len(grupos)}
    }

def palabras_mas_frecuentes(
    texto: str,
    top_n: int = 10,
    usar_lemmas: bool = True,
    excluir_stopwords: bool = True,
    min_len: int = 2,
    lang_code: str | None = None
) -> Dict[str, Any]:
    """
    Top-N palabras más frecuentes (ignorando stopwords y mínima longitud).
    Multi-idioma con spaCy; fallback a tokenización básica si no hay modelo.
    """
    if not lang_code:
        lang_code = detectar_idioma(texto)["codigo"]
    nlp = get_nlp(lang_code)

    if nlp is None:
        tokens = re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+", texto)
        tokens = [_strip_accents(t).lower() for t in tokens if len(t) >= min_len]
        cnt = Counter(tokens)
        top = cnt.most_common(top_n)
        return {
            "tipo": "palabras_mas_frecuentes",
            "idioma": lang_code,
            "top": [{"palabra": w, "conteo": c} for w, c in top],
            "metodo": "fallback_basico"
        }

    doc = nlp(texto)
    stopwords = nlp.Defaults.stop_words if excluir_stopwords else set()

    def valido(t) -> bool:
        if not t.is_alpha:
            return False
        if excluir_stopwords and t.text.lower() in stopwords:
            return False
        if len(t.text) < min_len:
            return False
        return True

    tokens = [t for t in doc if valido(t)]
    items = [t.lemma_.lower() if usar_lemmas else t.text.lower() for t in tokens]
    items = [_strip_accents(x) for x in items]  # normaliza acentos
    cnt = Counter(items)
    top = cnt.most_common(top_n)

    return {
        "tipo": "palabras_mas_frecuentes",
        "idioma": lang_code,
        "top": [{"palabra": w, "conteo": c} for w, c in top],
        "config": {"top_n": top_n, "usar_lemmas": usar_lemmas,
                   "excluir_stopwords": excluir_stopwords, "min_len": min_len},
        "total_tokens_considerados": len(items),
        "metodo": "spacy"
    }

# =========================
# Detección de temas (zero-shot + fallback)
# =========================

@lru_cache(maxsize=1)
def _get_zeroshot_pipeline():
    """Pipeline de zero-shot multilingüe; None si no disponible."""
    if pipeline is None:
        return None
    try:
        return pipeline("zero-shot-classification",
                        model="joeddav/xlm-roberta-large-xnli", device=-1)
    except Exception:  # pragma: no cover
        return None

def deteccion_temas(
    texto: str,
    etiquetas: List[str] | None = None,
    multi_label: bool = True,
    top_k: int = 3,
    lang_code: str | None = None
) -> Dict[str, Any]:
    if etiquetas is None:
        etiquetas = [
            "amor", "desamor", "tristeza", "nostalgia",
            "fiesta", "baile", "dinero", "éxito",
            "protesta", "rebelión", "amistad",
            "autoestima", "espiritualidad", "violencia"
        ]

    clf = _get_zeroshot_pipeline()
    if clf is not None:
        hyp = "Este texto es sobre {}."
        res = clf(texto, candidate_labels=etiquetas,
                  multi_label=multi_label, hypothesis_template=hyp)
        pares = list(zip(res["labels"], res["scores"]))
        pares = sorted(pares, key=lambda x: x[1], reverse=True)[:top_k]
        top = [{"tema": lab, "score": round(float(score) * 100, 1)}
               for lab, score in pares]
        return {"tipo": "deteccion_temas", "metodo": "zero_shot",
                "top": top, "etiquetas_consideradas": etiquetas}

    # Fallback por palabras clave
    keyword_map = {
        "amor": {"amor", "querer", "te amo", "love"},
        "desamor": {"olvido", "adiós", "ruptura", "heartbreak", "goodbye"},
        "tristeza": {"triste", "lloro", "lágrimas", "sad"},
        "nostalgia": {"recuerdo", "ayer", "añoro", "nostalgia"},
        "fiesta": {"fiesta", "party", "noche", "disco"},
        "baile": {"baila", "bailar", "dance"},
        "dinero": {"dinero", "money", "cash"},
        "éxito": {"fama", "éxito", "gloria", "win"},
        "protesta": {"protesta", "rebelión", "revolución", "resistencia"},
        "amistad": {"amigo", "amigos", "friend"},
        "autoestima": {"fuerte", "vencer", "valiente", "empoderado"},
        "espiritualidad": {"alma", "fe", "dios", "oración", "heaven"},
        "violencia": {"pelea", "guerra", "golpe", "arma", "violence"}
    }
    texto_low = texto.lower()
    scores = []
    for tema in etiquetas:
        kws = keyword_map.get(tema, set())
        s = sum(texto_low.count(k) for k in kws)
        scores.append((tema, s))
    scores.sort(key=lambda x: x[1], reverse=True)
    top = [{"tema": t, "score": float(c)} for t, c in scores[:top_k]]
    return {"tipo": "deteccion_temas", "metodo": "keywords",
            "top": top, "etiquetas_consideradas": etiquetas}

# =========================
# Metáforas (heurístico)
# =========================

def deteccion_metaforas(texto: str, lang_code: str | None = None) -> Dict[str, Any]:
    """
    Heurístico: símiles y patrones 'X de Y' (p.ej., 'corazón de piedra').
    """
    if not lang_code:
        lang_code = detectar_idioma(texto)["codigo"]

    lines_raw = texto.splitlines()
    lines = [_normalize_line(ln).lower() for ln in lines_raw]

    patterns: List[str]
    if lang_code == "es":
        patterns = [r"\bcomo\s+(un|una|el|la)\b", r"\btan\s+\w+\s+como\b"]
    elif lang_code == "en":
        patterns = [r"\blike\s+(a|an|the)\b", r"\bas\s+\w+\s+as\b"]
    elif lang_code == "fr":
        patterns = [r"\bcomme\s+(un|une|le|la)\b", r"\baussi\s+\w+\s+que\b"]
    elif lang_code == "pt":
        patterns = [r"\bcomo\s+(um|uma|o|a)\b", r"\btão\s+\w+\s+como\b"]
    elif lang_code == "de":
        patterns = [r"\bwie\s+(ein|eine|der|die|das)\b", r"\bso\s+\w+\s+wie\b"]
    else:
        patterns = [r"\bcomo\s+(un|una|el|la)\b", r"\btan\s+\w+\s+como\b"]

    similes = []
    for i, ln in enumerate(lines):
        for pat in patterns:
            if re.search(pat, ln):
                similes.append({"linea": lines_raw[i], "indice": i})
                break

    cabezas_meta = {"corazón", "alma", "vida", "sueño", "fuego", "hielo",
                    "cielo", "infierno", "mar", "tormenta", "luz", "sombra"}
    genit = []
    pat_de = re.compile(r"\b([a-záéíóúüñ]+)\s+de\s+([a-záéíóúüñ]+)\b", re.IGNORECASE)
    for i, ln in enumerate(lines):
        for m in pat_de.finditer(ln):
            head = m.group(1)
            if head in cabezas_meta:
                genit.append({"linea": lines_raw[i], "indice": i, "expresion": m.group(0)})
                break

    return {
        "tipo": "deteccion_metaforas",
        "idioma": lang_code,
        "similes": similes[:10],
        "metaforas_nominales": genit[:10],
        "conteos": {"similes": len(similes), "metaforas_nominales": len(genit),
                    "total": len(similes) + len(genit)},
        "nota": "Heurístico; se centra en símiles ('como/like/comme') y patrones 'X de Y' comunes."
    }

# =========================
# NER multi-idioma con spaCy
# =========================

def reconocimiento_entidades(texto: str, lang_code: str | None = None, top_n: int = 8) -> Dict[str, Any]:
    if not lang_code:
        lang_code = detectar_idioma(texto)["codigo"]
    nlp = get_nlp(lang_code) or get_nlp("es") or spacy.load("es_core_news_sm")
    doc = nlp(texto)

    por_tipo: Dict[str, Counter] = defaultdict(Counter)
    for ent in doc.ents:
        label = ent.label_
        txt = ent.text.strip()
        if txt:
            por_tipo[label][txt] += 1

    salida: Dict[str, Any] = {}
    total_entidades = 0
    for label, counter in por_tipo.items():
        total = sum(counter.values())
        total_entidades += total
        top = counter.most_common(top_n)
        salida[label] = {
            "etiqueta_humana": _LABEL_HUMANA.get(label, label),
            "total": total,
            "top": [{"texto": t, "conteo": c} for t, c in top]
        }

    return {"tipo": "reconocimiento_entidades",
            "idioma": lang_code,
            "total_entidades": total_entidades,
            "por_tipo": salida}
