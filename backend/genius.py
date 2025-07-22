import requests
from bs4 import BeautifulSoup

ACCESS_TOKEN = "LsnmCehh5fc2ub62tE2fxg2i-HNsBgzODgRxkptk4xY4OeJ4ycDJnBPy9bihYRfF"
HEADERS = {"Authorization": f"Bearer {ACCESS_TOKEN}"}

def buscar_canciones(query, max_results=5):
    url = "https://api.genius.com/search"
    params = {"q": query}
    response = requests.get(url, headers=HEADERS, params=params)

    if response.status_code != 200:
        return []

    hits = response.json()["response"]["hits"]
    resultados = []
    for hit in hits[:max_results]:
        resultados.append({
            "titulo": hit["result"]["title"],
            "artista": hit["result"]["primary_artist"]["name"],
            "url": hit["result"]["url"],
            "imagen": hit["result"]["song_art_image_url"]  # ✅ imagen añadida
        })

    return resultados

def obtener_letra_desde_url(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.text, "html.parser")

    lyrics_blocks = soup.find_all("div", class_=lambda value: value and "Lyrics__Container" in value)

    if not lyrics_blocks:
        return "Letra no encontrada."

    raw_lines = []
    for block in lyrics_blocks:
        raw_lines.extend(block.get_text(separator="\n").split("\n"))

    import re
    clean_lines = []
    for line in raw_lines:
        line = line.strip()
        if not line:
            clean_lines.append("")
            continue
        if re.match(r"\[.*?\]", line):
            continue
        if "lyrics" in line.lower() and len(line.split()) <= 3:
            continue
        if any(p in line.lower() for p in ["contributors", "translations", "read more"]):
            continue
        if re.match(r"^[\w\s\-–—\(\)]+$", line, flags=re.UNICODE) and len(line.split()) <= 3:
            continue
        clean_lines.append(line)

    start_index = 0
    for i, line in enumerate(clean_lines):
        if re.match(r"^[A-Z][^A-Z]+[a-z]{2,}", line) and not line.endswith(":"):
            start_index = i
            break

    letra_final = clean_lines[start_index:]

    resultado = []
    for i, line in enumerate(letra_final):
        if line == "" and (i == 0 or letra_final[i - 1] == ""):
            continue
        resultado.append(line)

    return "\n".join(resultado).strip()
