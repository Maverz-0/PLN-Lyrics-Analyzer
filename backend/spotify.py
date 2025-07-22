import requests
import base64

CLIENT_ID = "693e9edd43af4d088d7a9ce796a30aaf"
CLIENT_SECRET = "d980da09d62f42e086252844664028b7"

def obtener_token():
    url = "https://accounts.spotify.com/api/token"
    headers = {
        "Authorization": "Basic " + base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()
    }
    data = {
        "grant_type": "client_credentials"
    }

    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        token = response.json()["access_token"]
        print("🎟️ Token obtenido correctamente.")
        return token
    else:
        print("❌ Error al obtener token:")
        print("Código de estado:", response.status_code)
        print("Respuesta:", response.text)
        return None

def obtener_preview_url(titulo_cancion, artista):
    print(f"\n🎵 Buscando preview para: {titulo_cancion} - {artista}")
    token = obtener_token()
    if not token:
        print("❌ No se pudo obtener token de Spotify.")
        return None

    headers = {
        "Authorization": f"Bearer {token}"
    }

    query = f"{titulo_cancion} {artista}"
    url = f"https://api.spotify.com/v1/search?q={requests.utils.quote(query)}&type=track&limit=1"
    print("🔍 URL de búsqueda:", url)

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print("❌ Error al buscar la canción en Spotify.")
        print("Código:", response.status_code)
        print("Respuesta:", response.text)
        return None

    data = response.json()
    print("📦 Respuesta completa de Spotify:", data)

    items = data.get("tracks", {}).get("items", [])
    if not items:
        print("⚠️ No se encontraron resultados en Spotify.")
        return None

    track = items[0]
    preview_url = track.get("preview_url")

    if preview_url:
        print("✅ Preview URL encontrada:", preview_url)
    else:
        print("⚠️ La canción no tiene preview disponible.")
    
    return preview_url
