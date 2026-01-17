# -*- coding: utf-8 -*-
import os
import requests
from bs4 import BeautifulSoup
import re
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env (si existe)
load_dotenv()

# === API Genius (búsqueda) ===
# Leemos el token de la variable de entorno
ACCESS_TOKEN = os.getenv("GENIUS_ACCESS_TOKEN")

# Comprobación de seguridad para evitar errores confusos si falta el token
if not ACCESS_TOKEN:
    raise ValueError("Error: No se encontró la variable de entorno 'GENIUS_ACCESS_TOKEN'. Asegúrate de tener el archivo .env en local o la variable configurada en Coolify.")

HEADERS = {"Authorization": f"Bearer {ACCESS_TOKEN}"}

# Cabeceras tipo navegador para la página HTML
BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
}

def buscar_canciones(query, max_results=5):
    url = "https://api.genius.com/search"
    params = {"q": query}
    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=10)
    except Exception:
        return []

    if response.status_code != 200:
        return []

    hits = response.json()["response"]["hits"]
    resultados = []
    for hit in hits[:max_results]:
        resultados.append({
            "titulo": hit["result"]["title"],
            "artista": hit["result"]["primary_artist"]["name"],
            "url": hit["result"]["url"],
            "imagen": hit["result"]["song_art_image_url"]
        })
    return resultados


# =========================
#      UTILIDADES
# =========================

_RE_SECTION_FULL = re.compile(r"^\[.*?\]$")  # línea que ES solo [Sección]
_RE_HEADING_LYRICS = re.compile(r".*\blyrics\b", flags=re.I)

def _extract_lyrics_containers(soup: BeautifulSoup):
    """Contenedores reales de letra."""
    containers = soup.select('div[data-lyrics-container="true"]')
    if containers:
        return containers
    return soup.find_all("div", class_=lambda v: v and "Lyrics__Container" in v)

def _remove_split_section_blocks(lines):
    """
    Elimina bloques de sección que vienen en varias líneas:
      "[Verse 1: Playboi Carti &"  /  "]"
    y también las secciones en una sola línea "[Chorus]".
    ¡IMPORTANTE!: se ejecuta ANTES de cualquier filtrado por línea.
    """
    out = []
    inside = False
    for ln in lines:
        s = ln.strip()
        # Sección completa en una línea
        if not inside and _RE_SECTION_FULL.match(s):
            continue
        # Inicio de bloque partido
        if not inside and s.startswith("[") and not s.endswith("]"):
            inside = True
            continue
        if inside:
            # fin de bloque partido
            if "]" in s:
                inside = False
            continue
        out.append(ln)
    return out

def _drop_line(line: str) -> bool:
    """Filtra ruido ajeno a la letra (por línea). ¡Ojo: aquí ya NO borramos ']' sueltas!"""
    low = line.lower().strip()

    # Heading corto tipo “Title Lyrics”
    if _RE_HEADING_LYRICS.match(line) and len(line.split()) <= 5:
        return True

    # Bloques UI/promoción
    bad_fragments = [
        "you might also like",
        "embed", "contributors", "translations",
        "read more", "track info", "about", "credits",
        "produced by", "written by",
        "lyrics powered by",
        "see ",       # “See The Weeknd Live”
        "ver a ", "ver  traducción", "traducción al",
    ]
    if any(k in low for k in bad_fragments):
        return True

    # Solo números (anclas)
    if re.fullmatch(r"\d{1,3}", low):
        return True

    # Títulos sueltos genéricos muy cortos
    if re.match(r"^[\w\s\-–—\(\)]+$", line, flags=re.UNICODE) and len(line.split()) <= 3:
        return True

    return False

def _looks_like_prose(line: str) -> bool:
    """Heurística de prosa tipo 'About'."""
    s = line.strip()
    if len(s) >= 120 and s.endswith((".", "…", "?", "!")):
        return True
    if s.count(".") >= 1 and len(s) >= 110:
        return True
    punct = sum(1 for ch in s if ch in ".,;:!?")
    if len(s) >= 100 and punct / max(1, len(s)) > 0.04:
        return True
    return False

def _looks_like_verse(line: str) -> bool:
    """Señales de verso: moderado, poca puntuación, sin punto final."""
    s = line.strip()
    if not s:
        return False
    if len(s) > 120:
        return False
    if s.endswith((".", "…")):
        return False
    punct = sum(1 for ch in s if ch in ".,;:!?")
    return punct <= 2

def _collapse_blank_lines(lines):
    """Quita líneas vacías duplicadas y recorta extremos."""
    out = []
    prev_blank = False
    for ln in lines:
        is_blank = (ln == "")
        if is_blank and prev_blank:
            continue
        out.append(ln)
        prev_blank = is_blank
    while out and out[0] == "":
        out.pop(0)
    while out and out[-1] == "":
        out.pop()
    return out

def _fix_dangling_parentheses(lines):
    """
    Une líneas que terminan con '(' con ad-libs siguientes y cierra ')'.
    No elimina contenido: solo fusiona 1–3 líneas cortas.
    """
    out = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.strip().endswith("("):
            collected = []
            j = i + 1
            while j < len(lines):
                nxt = lines[j].strip()
                if not nxt:
                    break
                if nxt.startswith("["):
                    break
                if len(nxt) > 50:
                    break
                collected.append(nxt)
                if nxt.endswith(")") or ")" in nxt:
                    j += 1
                    break
                if len(collected) >= 3:
                    break
                j += 1
            if collected:
                joined = " ".join(collected)
                if ")" not in joined:
                    joined = joined + ")"
                out.append(ln.rstrip() + " " + joined)
                i = j
                continue
            else:
                # no había nada que pegar: quitamos el '(' colgante para no contaminar análisis
                out.append(ln.rstrip(" ("))
                i += 1
                continue
        else:
            out.append(ln)
            i += 1
    return out


# =========================
#    EXTRAER LA LETRA
# =========================

def obtener_letra_desde_url(url: str) -> str:
    try:
        page = requests.get(url, headers=BROWSER_HEADERS, timeout=12)
    except Exception:
        return "Letra no encontrada."

    if page.status_code != 200:
        return "Letra no encontrada."

    soup = BeautifulSoup(page.text, "html.parser")
    containers = _extract_lyrics_containers(soup)
    if not containers:
        return "Letra no encontrada."

    # 1) Extraer texto de contenedores de letra (sin limpiar aún)
    raw_lines = []
    for c in containers:
        block_text = c.get_text(separator="\n", strip=False)
        raw_lines.extend(block_text.split("\n"))

    # 2) ELIMINAR PRIMERO los bloques de sección partidos (para no depender de la línea ']' después)
    no_sections = _remove_split_section_blocks(raw_lines)

    # 3) Limpieza línea a línea (ya sin secciones partidas)
    cleaned = []
    for line in no_sections:
        line = (line or "").replace("\xa0", " ").strip()
        if not line:
            cleaned.append("")  # preserva estrofas
            continue
        if _drop_line(line):
            continue
        cleaned.append(line)

    # 4) Quitar prosa "About" al principio, hasta primer verso claro
    i = 0
    while i < len(cleaned) and (cleaned[i] == "" or _looks_like_prose(cleaned[i])):
        i += 1
    if i < len(cleaned) and (_RE_HEADING_LYRICS.search(cleaned[i]) and len(cleaned[i].split()) <= 5):
        i += 1
    main = cleaned[i:] if i < len(cleaned) else []

    if main:
        j = 0
        while j < len(main) and not _looks_like_verse(main[j]):
            j += 1
        main = main[j:]

    # 5) Arreglar paréntesis colgantes tipo "… ("
    main = _fix_dangling_parentheses(main)

    # 6) Compactar vacíos y recortar
    main = _collapse_blank_lines(main)

    if not main:
        return "Letra no encontrada."

    return "\n".join(main).strip()