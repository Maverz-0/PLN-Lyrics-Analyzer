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
  NER multi-idioma con spaCy
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


# Fonética EN (opcional)
try:
    import pronouncing  # usa CMUdict para inglés
    _HAS_PRONOUNCING = True
except Exception:
    _HAS_PRONOUNCING = False


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
# Sentimiento PRO (Transformers) — robusto (low-level con fallback)
# =========================
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

HF_CACHE = os.path.join(os.path.dirname(__file__), ".hf_cache")
os.environ.setdefault("HF_HOME", HF_CACHE)

# Modelo principal (3 clases: neg/neu/pos)
SENT_MODEL_ID = os.getenv("SENT_MODEL_ID", "cardiffnlp/twitter-xlm-roberta-base-sentiment")
SENT_MODEL_LOCAL = os.getenv("SENT_MODEL_LOCAL", "")  # ruta en disco si lo tienes

# Fallback (5 estrellas)
STAR_MODEL_ID = os.getenv("STAR_MODEL_ID", "nlptown/bert-base-multilingual-uncased-sentiment")
STAR_MODEL_LOCAL = os.getenv("STAR_MODEL_LOCAL", "")

# Estados (carga perezosa)
_SENT_TOKENIZER = None
_SENT_MODEL = None
_STAR_TOKENIZER = None
_STAR_MODEL = None

def _get_sent_components():
    """Carga tokenizer y modelo (principal) en modo evaluación."""
    global _SENT_TOKENIZER, _SENT_MODEL
    if _SENT_TOKENIZER is not None and _SENT_MODEL is not None:
        return _SENT_TOKENIZER, _SENT_MODEL
    model_ref = SENT_MODEL_LOCAL if SENT_MODEL_LOCAL else SENT_MODEL_ID
    _SENT_TOKENIZER = AutoTokenizer.from_pretrained(model_ref)
    _SENT_MODEL = AutoModelForSequenceClassification.from_pretrained(model_ref)
    _SENT_MODEL.eval()
    return _SENT_TOKENIZER, _SENT_MODEL

def _get_star_components():
    """Carga tokenizer y modelo (fallback 5⭐) en modo evaluación."""
    global _STAR_TOKENIZER, _STAR_MODEL
    if _STAR_TOKENIZER is not None and _STAR_MODEL is not None:
        return _STAR_TOKENIZER, _STAR_MODEL
    model_ref = STAR_MODEL_LOCAL if STAR_MODEL_LOCAL else STAR_MODEL_ID
    _STAR_TOKENIZER = AutoTokenizer.from_pretrained(model_ref)
    _STAR_MODEL = AutoModelForSequenceClassification.from_pretrained(model_ref)
    _STAR_MODEL.eval()
    return _STAR_TOKENIZER, _STAR_MODEL

def _get_label_indices(model):
    """Devuelve índices (neg, neu, pos) desde id2label; fallback 0/1/2 (Cardiff)."""
    cfg = getattr(model, "config", None)
    id2label = getattr(cfg, "id2label", None)
    neg = neu = pos = None
    if isinstance(id2label, dict):
        for i, lab in id2label.items():
            l = str(lab).lower()
            if "neg" in l:
                neg = int(i)
            elif "neu" in l:
                neu = int(i)
            elif "pos" in l:
                pos = int(i)
    # Fallback típico Cardiff: 0=neg, 1=neu, 2=pos
    if neg is None or neu is None or pos is None:
        neg = 0 if neg is None else neg
        neu = 1 if neu is None else neu
        pos = 2 if pos is None else pos
    return neg, neu, pos

@torch.no_grad()
def _infer_cardiff_probs(texts: List[str], batch_size: int = 16, max_length: int = 256):
    """
    Devuelve lista de tuplas (p_neg, p_neu, p_pos) por texto con el modelo Cardiff.
    """
    tok, model = _get_sent_components()
    neg_idx, neu_idx, pos_idx = _get_label_indices(model)

    out: List[Tuple[float, float, float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tok(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        logits = model(**enc).logits  # [B, 3]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        for p in probs:
            p_neg = float(p[neg_idx]); p_neu = float(p[neu_idx]); p_pos = float(p[pos_idx])
            s = p_neg + p_neu + p_pos
            if s > 0:
                p_neg, p_neu, p_pos = p_neg/s, p_neu/s, p_pos/s
            out.append((p_neg, p_neu, p_pos))
    return out

@torch.no_grad()
def _infer_star_probs(texts: List[str], batch_size: int = 16, max_length: int = 256):
    """
    Devuelve lista de tuplas (p1,p2,p3,p4,p5) por texto con el modelo 5⭐.
    """
    tok, model = _get_star_components()
    out: List[Tuple[float, float, float, float, float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tok(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        logits = model(**enc).logits  # [B, 5]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        for p in probs:
            p = [float(x) for x in p]
            s = sum(p)
            if s > 0:
                p = [x/s for x in p]
            out.append(tuple(p))  # (p1,...,p5)
    return out

def _agg_weighted(values: List[Tuple[float, int]]) -> float:
    num = sum(v * w for v, w in values)
    den = sum(w for _, w in values) or 1
    return num / den

def analizar_sentimiento_pro(texto: str) -> Dict[str, Any]:
    """
    Sentimiento [-1,1] con CardiffNLP (XLM-R). Si devuelve neutro “puro”,
    usa fallback 5⭐ (NLPTown) para estimar el signo.
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

    # Troceo por caracteres (conserva saltos)
    lines = _clean_lines(texto)
    chunks = _chunk_text(lines, max_chars=320)
    if not chunks:
        return base

    # --- 1) Intento con Cardiff (3-way) ---
    try:
        probs = _infer_cardiff_probs(chunks, batch_size=16, max_length=256)  # [(p_neg,p_neu,p_pos), ...]
        weights = [max(1, len(c)) for c in chunks]

        scores_w: List[Tuple[float, int]] = []
        pos_w: List[Tuple[float, int]] = []
        neu_w: List[Tuple[float, int]] = []
        neg_w: List[Tuple[float, int]] = []
        segs: List[Dict[str, Any]] = []

        for ch, (p_neg, p_neu, p_pos), w in zip(chunks, probs, weights):
            s = p_pos - p_neg
            scores_w.append((s, w))
            pos_w.append((p_pos, w))
            neu_w.append((p_neu, w))
            neg_w.append((p_neg, w))
            segs.append({
                "texto": ch if len(ch) <= 140 else ch[:137] + "…",
                "score": round(s, 3),
                "pos": round(p_pos, 3),
                "neu": round(p_neu, 3),
                "neg": round(p_neg, 3)
            })

        g_score = _agg_weighted(scores_w)
        g_pos = _agg_weighted(pos_w)
        g_neu = _agg_weighted(neu_w)
        g_neg = _agg_weighted(neg_w)

        # ¿Resultado sospechosamente neutro puro?
        if g_neu >= 0.98 and g_pos <= 0.01 and g_neg <= 0.01:
            # --- 2) Fallback 5⭐ ---
            try:
                star_probs = _infer_star_probs(chunks, batch_size=16, max_length=256)  # [(p1..p5), ...]
                # Mapeo a distribución negativa/neutral/positiva
                # - negativo: p1+p2
                # - neutral : p3
                # - positivo: p4+p5
                pos2_w: List[Tuple[float, int]] = []
                neu2_w: List[Tuple[float, int]] = []
                neg2_w: List[Tuple[float, int]] = []
                segs2: List[Dict[str, Any]] = []
                scores2_w: List[Tuple[float, int]] = []
                for ch, p, w in zip(chunks, star_probs, weights):
                    p1, p2, p3, p4, p5 = p
                    p_pos2 = p4 + p5
                    p_neg2 = p1 + p2
                    p_neu2 = p3
                    s2 = ( (1* p5 + 0.5* p4) - (1* p1 + 0.5* p2) )  # aproximación signo
                    pos2_w.append((p_pos2, w))
                    neu2_w.append((p_neu2, w))
                    neg2_w.append((p_neg2, w))
                    scores2_w.append((s2, w))
                    segs2.append({
                        "texto": ch if len(ch) <= 140 else ch[:137] + "…",
                        "score": round(s2, 3),
                        "pos": round(p_pos2, 3),
                        "neu": round(p_neu2, 3),
                        "neg": round(p_neg2, 3)
                    })
                g_score2 = _agg_weighted(scores2_w)
                g_pos2 = _agg_weighted(pos2_w)
                g_neu2 = _agg_weighted(neu2_w)
                g_neg2 = _agg_weighted(neg2_w)
                resumen2 = "positivo" if g_score2 > 0.2 else ("negativo" if g_score2 < -0.2 else "neutro")
                return {
                    **base,
                    "modelo": (SENT_MODEL_LOCAL or SENT_MODEL_ID) + " + fallback:" + (STAR_MODEL_LOCAL or STAR_MODEL_ID),
                    "score": round(g_score2, 3),
                    "resumen": resumen2,
                    "distribucion": {
                        "positivo": int(round(g_pos2 * 100)),
                        "neutral": int(round(g_neu2 * 100)),
                        "negativo": int(round(g_neg2 * 100)),
                    },
                    "segmentos_top": {
                        "positivos": sorted([s for s in segs2 if s["score"] > 0.1],
                                            key=lambda x: x["score"], reverse=True)[:2],
                        "negativos": sorted([s for s in segs2 if s["score"] < -0.1],
                                            key=lambda x: x["score"])[:2]
                    },
                    "total_segmentos": len(chunks),
                    "warning": "Cardiff devolvió neutro casi puro; usando fallback 5⭐."
                }
            except Exception as e2:
                # Si el fallback también falla, devolvemos el de Cardiff con aviso
                pass

        resumen = "positivo" if g_score > 0.2 else ("negativo" if g_score < -0.2 else "neutro")
        return {
            **base,
            "score": round(g_score, 3),
            "resumen": resumen,
            "distribucion": {
                "positivo": int(round(g_pos * 100)),
                "neutral": int(round(g_neu * 100)),
                "negativo": int(round(g_neg * 100)),
            },
            "segmentos_top": {
                "positivos": sorted([s for s in segs if s["score"] > 0.1],
                                    key=lambda x: x["score"], reverse=True)[:2],
                "negativos": sorted([s for s in segs if s["score"] < -0.1],
                                    key=lambda x: x["score"])[:2]
            },
            "total_segmentos": len(chunks)
        }
    except Exception as e:
        return {**base, "warning": f"Error en inferencia: {e}"}







# =========================
# Sentimiento SIMPLE (lexicón + reglas) — sin dependencias pesadas
# =========================
import math

# --- Léxicos mínimos por idioma (puedes ampliar cuando quieras) ---
_POS_LEX = {
    "en": {
        "love","lovely","happy","good","great","amazing","wonderful","like","beautiful","smile",
        "joy","blessed","awesome","best","bright","sweet","cool","peace","hope","calm","safe",
        "win","free","alive","shine","light","strong"
    },
    "es": {
        "amor","feliz","bueno","genial","increible","increíble","maravilloso","gusta","bonito","sonrisa",
        "alegria","alegría","bendecido","mejor","dulce","paz","esperanza","calma","seguro","brillar",
        "luz","fuerte","vivo","triunfo","ganar","libre"
    },
    "fr": {
        "amour","heureux","bien","génial","genial","incroyable","merveilleux","joli","sourire","joie",
        "béni","meilleur","doux","paix","espoir","calme","sûr","lumière","fort","libre","gagner"
    },
    "pt": {
        "amor","feliz","bom","ótimo","otimo","incrível","incrivel","maravilhoso","gosto","bonito","sorriso",
        "alegria","abençoado","melhor","doce","paz","esperança","calma","seguro","brilhar","luz","forte","livre","vencer"
    },
    "it": {
        "amore","felice","buono","grande","fantastico","fantástico","meraviglioso","bello","sorriso","gioia",
        "benedetto","migliore","dolce","pace","speranza","calma","sicuro","luce","forte","libero","vincere"
    },
    "de": {
        "liebe","glücklich","gut","großartig","grossartig","erstaunlich","wunderbar","schön","lächeln","freude",
        "gesegnet","beste","süß","suss","frieden","hoffnung","ruhig","sicher","licht","stark","frei","gewinnen"
    },
    "nl": {
        "liefde","blij","gelukkig","goed","geweldig","fantastisch","prachtig","mooi","glimlach","vreugde",
        "gezegend","beste","lief","vrede","hoop","rustig","veilig","licht","sterk","vrij","winnen"
    },
}

_NEG_LEX = {
    "en": {
        "hate","sad","bad","terrible","awful","horrible","pain","angry","cry","tears","lonely",
        "broken","kill","die","dead","hurt","dark","ugly","worst","sick","fear","afraid","anxious",
        "hell","damn","shit","fuck","alone"," sorrow","bleed"
    },
    "es": {
        "odio","triste","malo","terrible","horrible","dolor","enojado","llorar","lágrimas","lagrimas",
        "solo","solitario","rota","matar","morir","muerto","herida","oscuro","feo","peor","enfermo",
        "miedo","asustado","ansioso","infierno","maldito","sufrir","sangrar"
    },
    "fr": {
        "haine","triste","mauvais","terrible","horrible","douleur","fâché","fache","pleurer","larmes",
        "seul","brisé","tuer","mourir","mort","blessure","sombre","laid","pire","malade","peur",
        "effrayé","anxieux","enfer","maudit","souffrir","saigner"
    },
    "pt": {
        "ódio","odio","triste","mau","terrível","terrivel","horrível","horrivel","dor","raiva","chorar","lágrimas","lagrimas",
        "sozinho","partido","matar","morrer","morto","ferido","escuro","feio","pior","doente","medo","assustado",
        "ansioso","inferno","amaldiçoado","amaldiçoado","sofrer","sangrar"
    },
    "it": {
        "odio","triste","cattivo","terribile","orribile","dolore","arrabbiato","piangere","lacrime",
        "solo","rotto","uccidere","morire","morto","ferito","scuro","brutto","peggiore","malato",
        "paura","spaventato","ansioso","inferno","maledetto","soffrire","sanguinare"
    },
    "de": {
        "hass","traurig","schlecht","schrecklich","furchtbar","grauenhaft","schmerz","wütend","weinen","tränen","tranen",
        "einsam","gebrochen","töten","toeten","sterben","tot","verletz","dunkel","hässlich","haesslich","schlimmste",
        "krank","angst","fürchte","furchte","hell","verdammt","leiden","bluten"
    },
    "nl": {
        "haat","verdrietig","slecht","verschrikkelijk","afschuwelijk","pijn","boos","huilen","tranen",
        "eenzaam","gebroken","doden","sterven","dood","gewond","donker","lelijk","ergste","ziek",
        "angst","bang","vervloekt","lijden","bloeden"
    },
}

# Negadores / intensificadores / atenuadores
_NEGATORS = {
    "en": {"not","no","never","without","nobody","nothing","n't"},
    "es": {"no","nunca","jamás","jamas","sin","nadie","nada"},
    "fr": {"ne","pas","jamais","sans","rien","personne"},
    "pt": {"não","nao","nunca","jamais","sem","ninguém","ninguem","nada"},
    "it": {"non","mai","senza","nessuno","niente"},
    "de": {"nicht","nie","ohne","niemand","nichts","kein","keine","keinen","keinem","keiner","keines"},
    "nl": {"niet","nooit","zonder","niemand","niets","geen"}
}
_INTENSIFIERS = {
    "en": {"very","so","really","too","super","extremely","quite"},
    "es": {"muy","tan","super","re","extremadamente","bastante"},
    "fr": {"très","tres","tellement","vraiment","hyper","assez"},
    "pt": {"muito","tão","tanto","super","realmente","bastante"},
    "it": {"molto","tanto","davvero","super","abbastanza"},
    "de": {"sehr","wirklich","echt","ziemlich","mega"},
    "nl": {"erg","heel","echt","best","vrij","super"}
}
_DIMINISHERS = {
    "en": {"slightly","little","a little","a bit","kinda","sort","somewhat"},
    "es": {"un","poco","algo","apenas"},
    "fr": {"un","peu","légèrement","legerement"},
    "pt": {"um","pouco","ligeiramente"},
    "it": {"un","po","poco","leggermente"},
    "de": {"ein","bisschen","leicht"},
    "nl": {"een","beetje","lichtjes"}
}

_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ']+")

def _tokenize_simple(text: str) -> List[str]:
    # minúsculas y sin signos; mantenemos acentos para casarlas con el léxico
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]

def _lemma_tokens(text: str, lang_code: str) -> List[str]:
    """Si hay modelo spaCy del idioma, usamos lemmas; si no, tokens simples."""
    nlp = get_nlp(lang_code)
    if nlp is None:
        return _tokenize_simple(text)
    doc = nlp(text)
    return [t.lemma_.lower() for t in doc if t.is_alpha]

def analizar_sentimiento_simple(texto: str, lang_code: str | None = None) -> Dict[str, Any]:
    """
    Analiza sentimiento en [-1,1] con léxico + reglas:
    - suma (+1/-1) por palabra positiva/negativa
    - niega si hay negador en ventana previa (3 tokens)
    - intensifica (x1.5) / atenúa (x0.5)
    - pequeño boost por signos de exclamación
    """
    if not texto or not texto.strip():
        return {
            "tipo": "sentimiento",
            "modelo": "lexicon_simple_v1",
            "score": 0.0,
            "resumen": "neutro",
            "distribucion": {"positivo": 0, "neutral": 100, "negativo": 0},
            "segmentos_top": {"positivos": [], "negativos": []},
            "total_segmentos": 0
        }

    if not lang_code:
        lang_code = detectar_idioma(texto)["codigo"]
    lang = lang_code if lang_code in _POS_LEX else "en"  # fallback a EN

    pos_set = _POS_LEX.get(lang, _POS_LEX["en"])
    neg_set = _NEG_LEX.get(lang, _NEG_LEX["en"])
    negators = _NEGATORS.get(lang, _NEGATORS["en"])
    intens = _INTENSIFIERS.get(lang, _INTENSIFIERS["en"])
    dimins = _DIMINISHERS.get(lang, _DIMINISHERS["en"])

    toks = _lemma_tokens(texto, lang)

    S = 0.0           # suma firmada
    pos_sum = 0.0     # magnitud positiva
    neg_sum = 0.0     # magnitud negativa
    matches = 0       # palabras contabilizadas

    window = []       # últimas 3 palabras
    for w in toks:
        # ventana: últimas 3 tokens
        prev3 = window[-3:]

        # factores de refuerzo/atenuación
        boost = 1.0
        if any(p in intens for p in prev3):
            boost *= 1.5
        if any(p in dimins for p in prev3):
            boost *= 0.5

        # negación previa
        negated = any(p in negators for p in prev3)

        if w in pos_set:
            val = 1.0 * boost
            if negated: val *= -1.0
            S += val
            pos_sum += max(0.0, val)
            neg_sum += max(0.0, -val)
            matches += 1
        elif w in neg_set:
            val = -1.0 * boost
            if negated: val *= -1.0
            S += val
            pos_sum += max(0.0, val)
            neg_sum += max(0.0, -val)
            matches += 1

        window.append(w)

    # Boost suave por exclamaciones
    excls = texto.count("!")
    if excls:
        S *= min(1.0 + 0.05 * min(excls, 10), 1.3)

    # Normalización a [-1,1]
    if matches == 0:
        score = 0.0
        p_pos = 0.0; p_neg = 0.0; p_neu = 1.0
    else:
        score = math.tanh(S / (matches + 1.0))  # suave y acotado
        total_mag = pos_sum + neg_sum
        if total_mag > 0:
            p_pos = pos_sum / total_mag
            p_neg = neg_sum / total_mag
            p_neu = max(0.0, 1.0 - p_pos - p_neg)
        else:
            p_pos = 0.0; p_neg = 0.0; p_neu = 1.0

    resumen = "positivo" if score > 0.2 else ("negativo" if score < -0.2 else "neutro")

    return {
        "tipo": "sentimiento",
        "modelo": "lexicon_simple_v1",
        "score": round(float(score), 3),
        "resumen": resumen,
        "distribucion": {
            "positivo": int(round(p_pos * 100)),
            "neutral":  int(round(p_neu * 100)),
            "negativo": int(round(p_neg * 100)),
        },
        "segmentos_top": {"positivos": [], "negativos": []},  # opcional: podríamos puntuar líneas
        "total_segmentos": 1
    }









# =========================
# Análisis: morfosintaxis y léxico
# =========================

def porcentaje_adjetivos(texto: str, lang_code: str | None = None) -> Dict[str, Any]:
    if not lang_code:
        lang_code = detectar_idioma(texto)["codigo"]
    nlp = get_nlp(lang_code)

    if nlp is None:
        # No hacemos fallback a ES: avisamos del problema
        return {
            "tipo": "porcentaje_adjetivos",
            "porcentaje": None,
            "adjetivos": None,
            "total": None,
            "idioma": lang_code,
            "warning": (
                f"No hay modelo spaCy para '{lang_code}' instalado. "
                f"Instala: python -m spacy download {SPACY_MODELS.get(lang_code, '<no-disponible>')}"
            ),
            "metodo": "sin_modelo"
        }

    doc = nlp(texto)
    total_palabras = len([t for t in doc if t.is_alpha])
    adjetivos = len([t for t in doc if t.pos_ == "ADJ"])
    if total_palabras == 0:
        return {
            "tipo": "porcentaje_adjetivos", "porcentaje": 0, "adjetivos": 0, "total": 0,
            "idioma": lang_code, "metodo": "spacy"
        }
    porcentaje = round((adjetivos / total_palabras) * 100, 2)
    return {
        "tipo": "porcentaje_adjetivos",
        "porcentaje": porcentaje,
        "adjetivos": adjetivos,
        "total": total_palabras,
        "idioma": lang_code,
        "metodo": "spacy"
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

def _rhyme_key_soft(word_norm: str, vowels: set[str], min_tail: int = 2) -> str:
    """
    Clave 'suave' = desde la ÚLTIMA vocal (núcleo) hasta el final (coda).
    Ej.: acabar -> 'ar', aclarar -> 'ar', canción -> 'on', amor -> 'or'.
    Si el tail es muy corto (p.ej. 'a'), se descarta (evita sobre-agrupaciones).
    """
    # Busca la última vocal desde el final
    for i in range(len(word_norm) - 1, -1, -1):
        if word_norm[i] in vowels:
            tail = word_norm[i:]
            return tail if len(tail) >= min_tail else ""
    # si no hay vocal, caemos a los 2 últimos caracteres
    return word_norm[-min_tail:] if len(word_norm) >= min_tail else word_norm

def _en_rhyme_key_phonetic(word: str) -> str:
    """
    Clave fonética para inglés: desde la última vocal acentuada.
    Usa CMUdict vía 'pronouncing'. Fallback: "" si no hay entrada.
    """
    if not _HAS_PRONOUNCING:
        return ""
    w = re.sub(r"[^a-zA-Z’']+", "", word).lower()
    # normalizaciones típicas: posesivo, in' -> ing
    w = re.sub(r"(?:'s|’s)$", "", w)
    w = re.sub(r"in'$", "ing", w)

    phones = pronouncing.phones_for_word(w)
    if not phones and w.endswith("es"):
        phones = pronouncing.phones_for_word(w[:-2])
    if not phones and w.endswith("s"):
        phones = pronouncing.phones_for_word(w[:-1])
    if not phones and w.endswith("ing"):
        phones = pronouncing.phones_for_word(w[:-3])
    if not phones and w.endswith("ed"):
        phones = pronouncing.phones_for_word(w[:-2])
    if not phones and w.endswith("e"):
        phones = pronouncing.phones_for_word(w[:-1])

    if not phones:
        return ""

    ph = phones[0]
    try:
        rp = pronouncing.rhyming_part(ph)  # desde la última vocal acentuada
    except Exception:
        return ""
    # limpieza: quitamos dígitos de acento y espacios
    return re.sub(r"[0-2 ]", "", rp).strip()


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

def deteccion_rimas(
    texto: str,
    lang_code: str | None = None,
    tail_len: int = 3,
    prioridad: Tuple[str, ...] = None,
    min_tail_suave: int = 2
) -> Dict[str, Any]:
    """
    Detección de rimas por fin de verso:

      - 'fonetica'  (solo EN): última vocal acentuada + coda (CMUdict)
      - 'suave'     (multi-idioma): última vocal + coda (texto)
      - 'consonante': últimos N chars
      - 'asonante'  : vocales del tramo final

    Se ignoran paréntesis/ad-libs antes de procesar.
    """
    if not lang_code:
        lang_code = detectar_idioma(texto)["codigo"]
    _ = get_nlp(lang_code) or get_nlp("es") or spacy.load("es_core_news_sm")
    vowels = _lang_vowels(lang_code)

    # Prioridad por defecto: EN usa fonética primero; otros, suave primero
    if prioridad is None:
        if lang_code == "en" and _HAS_PRONOUNCING:
            prioridad = ("fonetica", "suave", "consonante", "asonante")
        else:
            prioridad = ("suave", "consonante", "asonante")

    raw_lines = [ln for ln in texto.splitlines()]
    pre = [_strip_parentheticals(ln) for ln in raw_lines]
    lines = [_normalize_line(ln) for ln in pre]

    idx_map = [i for i, ln in enumerate(lines) if ln]
    lines = [ln for ln in lines if ln]
    if not lines:
        return {"tipo": "deteccion_rimas", "idioma": lang_code,
                "num_versos": 0, "esquema": [], "grupos": [], "stats": {}}

    # Última palabra normalizada
    last_words = [_strip_accents(_last_word(ln)).lower() for ln in lines]

    # Claves
    keys_suave = [_rhyme_key_soft(w, vowels=vowels, min_tail=min_tail_suave) for w in last_words]
    keys_cons  = [_rhyme_key_consonant(w, tail_len=tail_len) for w in last_words]
    keys_asso  = [_rhyme_key_assonant(w, vowels=vowels) for w in last_words]
    keys_phon  = []
    if lang_code == "en" and _HAS_PRONOUNCING:
        keys_phon = [_en_rhyme_key_phonetic(w) for w in last_words]

    # Agrupación con prioridad
    group_map: Dict[int, str] = {}
    clusters: List[Dict[str, Any]] = []
    label_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    next_label_idx = 0
    used: set[int] = set()

    def _agrupar(keys: List[str], tipo: str):
        nonlocal next_label_idx
        buckets: Dict[str, List[int]] = defaultdict(list)
        for i, k in enumerate(keys):
            if i in used:
                continue
            buckets[k].append(i)
        for k, members in buckets.items():
            if k and len(members) >= 2:
                label = label_chars[next_label_idx % len(label_chars)]
                next_label_idx += 1
                for m in members:
                    group_map[m] = label
                    used.add(m)
                clusters.append({"tipo": tipo, "clave": k, "miembros": members})

    for step in prioridad:
        if step == "fonetica" and keys_phon:
            _agrupar(keys_phon, "fonetica")
        elif step == "suave":
            _agrupar(keys_suave, "suave")
        elif step == "consonante":
            _agrupar(keys_cons, "consonante")
        elif step == "asonante":
            _agrupar(keys_asso, "asonante")

    esquema = [group_map.get(i, "-") for i in range(len(lines))]

    versos_con_rima = sum(1 for e in esquema if e != "-")
    stats = {
        "versos": len(lines),
        "versos_con_rima": versos_con_rima,
        "porcentaje_con_rima": round(versos_con_rima * 100 / len(lines), 2),
        "num_grupos": len(clusters),
        "rimas_foneticas": sum(1 for c in clusters if c["tipo"] == "fonetica"),
        "rimas_suaves": sum(1 for c in clusters if c["tipo"] == "suave"),
        "rimas_consonantes": sum(1 for c in clusters if c["tipo"] == "consonante"),
        "rimas_asonantes": sum(1 for c in clusters if c["tipo"] == "asonante"),
    }

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
    Heurístico ampliado y más permisivo:
      - Símiles: 'como ...', 'tan ADJ como', 'like ...', 'as ADJ as', etc.
      - Copulares: 'soy/eres/es ...', 'I am/you are ...', 'mi corazón es ...', 'my heart is ...'
      - Genitivas: 'X de Y' / 'X of Y' (p.ej., 'corazón de piedra', 'heart of stone')
    Devuelve hasta 10 hallazgos por categoría.
    """
    if not lang_code:
        lang_code = detectar_idioma(texto)["codigo"]

    # --- Prepro: quitar paréntesis/ad-libs y normalizar línea a línea ---
    raw_lines = texto.splitlines()
    pre = [_strip_parentheticals(ln) for ln in raw_lines]
    lines = [_normalize_line(ln).lower() for ln in pre]

    # --- Listas semánticas para recortar falsos positivos ---
    # cabezas típicas de metáfora nominal (cosa poseída o núcleo)
    HEADS_ES = {"corazon", "corazón", "alma", "vida", "sangre", "piel", "voz", "mente",
                "suenos", "sueños", "fuego", "hielo", "luz", "sombra", "mar", "tormenta", "cielo", "infierno"}
    HEADS_EN = {"heart", "soul", "life", "blood", "skin", "voice", "mind", "dream",
                "fire", "ice", "light", "shadow", "sea", "storm", "heaven", "hell"}
    # materiales/abstractos muy usados en metáforas
    MAT_ES = {"piedra","acero","cristal","oro","plata","veneno","miel","sal","fuego","hielo","nieve","llama","sombra","luz"}
    MAT_EN = {"stone","steel","glass","gold","silver","poison","honey","salt","fire","ice","snow","flame","shadow","light"}
    # sustantivos-objetivo frecuentes en copulares: “eres mi sol”, “I am a mess”
    TARGET_ES = {"sol","luna","ancla","refugio","tormenta","tempestad","veneno","cura","luz","sombra",
                 "angel","ángel","demonio","desastre","abrigo","puerto","espina","flor","jaula","prision","prisión"}
    TARGET_EN = {"sun","moon","anchor","shelter","storm","poison","cure","light","shadow",
                 "angel","demon","mess","harbor","thorn","flower","cage","prison"}

    # --- Patrones por idioma ---
    patt_similes = []
    patt_cop = []
    patt_gen = []

    if lang_code == "es":
        # SÍMILES
        patt_similes = [
            r"\bcomo(?:\s+si)?\s+(?:un|una|el|la|los|las|mi|tu|su)\s+[a-záéíóúüñ]{2,}(?:\s+[a-záéíóúüñ]{2,}){0,3}\b",
            r"\btan\s+[a-záéíóúüñ]{2,}\s+como\b",
        ]
        # COPULARES (ser) con sustantivos frecuentes o posesivos
        patt_cop = [
            r"\b(?:soy|eres|es|somos|sois|son)\s+(?:un|una|el|la|mi|tu|su)?\s*[a-záéíóúüñ]{3,}\b",
            r"\bmi\s+corazon\s+es\s+(?:de\s+)?[a-záéíóúüñ]{3,}\b",
            r"\bmi\s+corazón\s+es\s+(?:de\s+)?[a-záéíóúüñ]{3,}\b",
        ]
        # GENITIVAS
        patt_gen = [
            r"\b([a-záéíóúüñ]+)\s+de\s+([a-záéíóúüñ]+)\b",
        ]
    elif lang_code == "en":
        patt_similes = [
            r"\blike\s+(?:a|an|the|my|your|his|her|our|their)\s+[a-z']{2,}(?:\s+[a-z']{2,}){0,3}\b",
            r"\bas\s+[a-z']{2,}\s+as\b",
        ]
        patt_cop = [
            r"\b(?:i am|i'm|you are|you're|he is|she is|it's|we are|we're|they are|they're)\s+(?:a|an|the|my|your|his|her|our|their)?\s*[a-z']{3,}\b",
            r"\bmy\s+heart\s+is\s+(?:of\s+)?[a-z']{3,}\b",
        ]
        patt_gen = [
            r"\b([a-z']+)\s+of\s+([a-z']+)\b",
        ]
    elif lang_code == "fr":
        patt_similes = [
            r"\bcomme\s+(?:un|une|le|la|mes|tes|ses|nos|vos|leurs)\s+[a-zàâçéèêëîïôûùüÿœ']{2,}(?:\s+[a-zàâçéèêëîïôûùüÿœ']{2,}){0,3}\b",
            r"\baussi\s+[a-zàâçéèêëîïôûùüÿœ']{2,}\s+que\b",
        ]
        patt_cop = [
            r"\b(?:je\s+suis|tu\s+es|il\s+est|elle\s+est|nous\s+sommes|vous\s+êtes|ils\s+sont|elles\s+sont)\s+(?:un|une|le|la|mon|ma|ton|ta|son|sa)?\s*[a-zàâçéèêëîïôûùüÿœ']{3,}\b",
        ]
        patt_gen = [
            r"\b([a-zàâçéèêëîïôûùüÿœ']+)\s+de\s+([a-zàâçéèêëîïôûùüÿœ']+)\b",
        ]
    elif lang_code == "pt":
        patt_similes = [
            r"\bcomo\s+(?:um|uma|o|a|meu|minha|teu|tua|seu|sua|nosso|nossa)\s+[a-záâãàçéêíïóôõúü']{2,}(?:\s+[a-záâãàçéêíïóôõúü']{2,}){0,3}\b",
            r"\btão\s+[a-záâãàçéêíïóôõúü']{2,}\s+como\b",
        ]
        patt_cop = [
            r"\b(?:sou|és|é|somos|sois|são)\s+(?:um|uma|o|a|meu|minha|teu|tua|seu|sua)?\s*[a-záâãàçéêíïóôõúü']{3,}\b",
        ]
        patt_gen = [
            r"\b([a-záâãàçéêíïóôõúü']+)\s+de\s+([a-záâãàçéêíïóôõúü']+)\b",
        ]
    elif lang_code == "de":
        patt_similes = [
            r"\bwie\s+(?:ein|eine|der|die|das|mein|dein|sein|ihr|unser|euer)\s+[a-zäöüß']{2,}(?:\s+[a-zäöüß']{2,}){0,3}\b",
            r"\bso\s+[a-zäöüß']{2,}\s+wie\b",
        ]
        patt_cop = [
            r"\b(?:ich\s+bin|du\s+bist|er\s+ist|sie\s+ist|wir\s+sind|ihr\s+seid|sie\s+sind)\s+(?:ein|eine|der|die|das|mein|dein|sein|ihr|unser|euer)?\s*[a-zäöüß']{3,}\b",
        ]
        patt_gen = [
            r"\b([a-zäöüß']+)\s+aus\s+([a-zäöüß']+)\b",  # 'aus Stahl', etc.
            r"\b([a-zäöüß']+)\s+von\s+([a-zäöüß']+)\b",
        ]
    else:
        # Por defecto, usa variantes ES/EN más comunes
        patt_similes = [
            r"\bcomo\s+(?:un|una|el|la|mi|tu|su)\s+[a-záéíóúüñ']{2,}(?:\s+[a-záéíóúüñ']{2,}){0,3}\b",
            r"\blike\s+(?:a|an|the|my|your)\s+[a-z']{2,}(?:\s+[a-z']{2,}){0,3}\b",
            r"\bas\s+[a-z']{2,}\s+as\b",
            r"\btan\s+[a-záéíóúüñ']{2,}\s+como\b",
        ]
        patt_cop = [
            r"\b(?:soy|eres|es|somos|son)\s+(?:un|una|el|la|mi|tu|su)?\s*[a-záéíóúüñ']{3,}\b",
            r"\b(?:i am|i'm|you are|you're)\s+(?:a|an|the|my|your)?\s*[a-z']{3,}\b",
            r"\bmi\s+corazon\s+es\s+(?:de\s+)?[a-záéíóúüñ']{3,}\b",
            r"\bmy\s+heart\s+is\s+(?:of\s+)?[a-z']{3,}\b",
        ]
        patt_gen = [
            r"\b([a-záéíóúüñ']+)\s+de\s+([a-záéíóúüñ']+)\b",
            r"\b([a-z']+)\s+of\s+([a-z']+)\b",
        ]

    # --- Búsqueda ---
    similes: List[Dict[str, Any]] = []
    copulares: List[Dict[str, Any]] = []
    genitivas: List[Dict[str, Any]] = []

    # Compila patrones
    sim_re = [re.compile(p, re.IGNORECASE) for p in patt_similes]
    cop_re = [re.compile(p, re.IGNORECASE) for p in patt_cop]
    gen_re = [re.compile(p, re.IGNORECASE) for p in patt_gen]

    # Helpers para filtros semánticos
    def es_target_es(w: str) -> bool:
        wn = _strip_accents(w)
        return wn in { _strip_accents(x) for x in TARGET_ES }
    def es_target_en(w: str) -> bool:
        return w in TARGET_EN
    def es_head_es(w: str) -> bool:
        wn = _strip_accents(w)
        return wn in { _strip_accents(x) for x in HEADS_ES }
    def es_head_en(w: str) -> bool:
        return w in HEADS_EN
    def es_mat_es(w: str) -> bool:
        wn = _strip_accents(w)
        return wn in { _strip_accents(x) for x in MAT_ES }
    def es_mat_en(w: str) -> bool:
        return w in MAT_EN

    for i, (raw, ln) in enumerate(zip(raw_lines, lines)):
        if not ln:
            continue

        # SÍMILES
        for rgx in sim_re:
            m = rgx.search(ln)
            if m:
                frag = m.group(0)
                similes.append({"linea": raw, "indice": i, "expresion": frag})
                break  # una por línea

        # COPULARES (filtramos con listas objetivo para evitar falsos positivos)
        for rgx in cop_re:
            m = rgx.search(ln)
            if not m:
                continue
            frag = m.group(0)
            ok = True
            # intenta usar la última palabra como “núcleo”
            tail = frag.split()[-1]
            if lang_code == "es":
                ok = es_target_es(tail) or tail.startswith(("mi","tu","su"))
            elif lang_code == "en":
                ok = es_target_en(tail) or tail in {"mine","yours","hers","his","ours","theirs"}
            # si no pasó filtro, no añadimos
            if ok:
                copulares.append({"linea": raw, "indice": i, "expresion": frag})
                break

        # GENITIVAS (X de Y / X of Y) con cabezas y materiales conocidos
        for rgx in gen_re:
            for m in rgx.finditer(ln):
                a, b = m.group(1), m.group(2)
                a2 = _strip_accents(a)
                b2 = _strip_accents(b)
                ok = False
                if lang_code == "es":
                    ok = (es_head_es(a) and es_mat_es(b)) or (a2 in {"corazon","corazón","alma"} and b2 not in {"ti","mi","tu"})
                elif lang_code == "en":
                    ok = (es_head_en(a) and es_mat_en(b)) or (a in {"heart","soul"} and b not in {"you","me"})
                else:
                    # por defecto, aplica reglas débiles
                    ok = len(a) >= 3 and len(b) >= 3 and a != b
                if ok:
                    genitivas.append({"linea": raw, "indice": i, "expresion": m.group(0)})

    # Limita resultados por categoría
    similes = similes[:10]
    copulares = copulares[:10]
    genitivas = genitivas[:10]

    return {
        "tipo": "deteccion_metaforas",
        "idioma": lang_code,
        "similes": similes,
        "metaforas_nominales": genitivas,
        "copulares": copulares,  # NUEVO: metáforas tipo 'X es Y'
        "conteos": {
            "similes": len(similes),
            "metaforas_nominales": len(genitivas),
            "copulares": len(copulares),
            "total": len(similes) + len(genitivas) + len(copulares)
        },
        "nota": "Heurístico ampliado. Puede dar falsos positivos/negativos; útil como detector aproximado."
    }


# =========================
# NER multi-idioma con spaCy
# =========================

# ==== Helpers para filtrar entidades ruidosas (ad-libs / onomatopeyas) ====

_ADLIBS = {
    "la","na","da","ba","pa","ra","tra",
    "oh","ah","eh","uh","ooh","aah","mm","mmm",
    "yeah","ya","ey","hey","yo","woah","whoa","huh","doo","sha"
}

_TOKEN_MINI_RE = re.compile(r"[a-záéíóúüñ]{1,3}", re.IGNORECASE)

def _is_repeated_short_syllables(text: str) -> bool:
    """
    True si el texto es básicamente una repetición de sílabas/onomatopeyas
    de 1–3 letras (p.ej., 'la la la', 'na-na-na', 'oh oh oh').
    """
    toks = _TOKEN_MINI_RE.findall(text.lower())
    if len(toks) < 3:
        return False
    uniq = set(toks)
    # caso típico: todo es la misma sílaba, o son 2 muy comunes de ad-lib
    return (
        len(uniq) == 1
        or (len(uniq) == 2 and all(u in _ADLIBS for u in uniq))
        or all(u in _ADLIBS for u in uniq)  # todo son ad-libs conocidos
    )

def _looks_like_onomatopoeia(ent_text: str) -> bool:
    s = ent_text.strip().lower()
    # una sola "palabra" y es ad-lib
    if s in _ADLIBS:
        return True
    # repetición de sílabas/ad-libs
    if _is_repeated_short_syllables(s):
        return True
    # variantes con comas/puntos: "la, la, la", "la-la-la"
    s_letters = re.sub(r"[^a-záéíóúüñ]+", " ", s).strip()
    if _is_repeated_short_syllables(s_letters):
        return True
    return False

def _should_drop_loc_like(ent_text: str, ent_label: str, doc_text: str) -> bool:
    """
    Reglas conservadoras para LOC/GPE:
    - descarta 'la'/'LA' suelta y repeticiones tipo 'la la la'
    - descarta 'LA' salvo que haya evidencia de 'Los Angeles/Los Ángeles' en el texto
    """
    if ent_label not in {"GPE", "LOC"}:
        return False
    t = ent_text.strip()
    tl = t.lower()

    # Repeticiones/onomatopeyas → fuera
    if _looks_like_onomatopoeia(t):
        return True

    # 'LA' ambiguo: solo lo permitimos si hay contexto explícito en el texto completo
    if t in {"LA", "La", "L.A.", "L. A."}:
        if re.search(r"\bLos\s+Angeles\b", doc_text, re.IGNORECASE) or \
           re.search(r"\bLos\s+Ángeles\b", doc_text, re.IGNORECASE):
            return False  # hay contexto → mantener
        return True  # sin contexto → descartar

    # 'la' minúscula suelta (artículo español) como lugar → descartar
    if tl == "la":
        return True

    return False


def reconocimiento_entidades(texto: str, lang_code: str | None = None, top_n: int = 8) -> Dict[str, Any]:
    if not lang_code:
        lang_code = detectar_idioma(texto)["codigo"]
    nlp = get_nlp(lang_code) or get_nlp("es") or spacy.load("es_core_news_sm")
    doc = nlp(texto)

    por_tipo: Dict[str, Counter] = defaultdict(Counter)

    for ent in doc.ents:
        etxt = ent.text.strip()
        if not etxt:
            continue

        # --- Filtro anti-ruido ---
        if _looks_like_onomatopoeia(etxt):
            continue
        if _should_drop_loc_like(etxt, ent.label_, doc.text):
            continue
        # (Opcional) si todos los tokens de la entidad son DET/INTJ, también descartar
        if all(t.pos_ in {"DET", "INTJ"} for t in ent):
            continue

        por_tipo[ent.label_][etxt] += 1

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

    return {
        "tipo": "reconocimiento_entidades",
        "idioma": lang_code,
        "total_entidades": total_entidades,
        "por_tipo": salida
    }

