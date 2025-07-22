import requests

def obtener_letra(artista, cancion):
    url = f"https://api.lyrics.ovh/v1/{artista}/{cancion}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        return data.get("lyrics", "Letra no encontrada.")
    else:
        return f"Error {response.status_code}: No se encontró la canción."
