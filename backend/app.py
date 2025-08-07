from flask import Flask, request, jsonify
from flask_cors import CORS
from genius import buscar_canciones, obtener_letra_desde_url
from analyzer import porcentaje_adjetivos
from spotify import obtener_preview_url



app = Flask(__name__)

# CORS: permitir todos los orígenes durante desarrollo (ajusta en producción)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/analizar', methods=['POST'])
def analizar():
    data = request.json
    artista = data.get('artista')
    cancion = data.get('cancion')
    tipo_analisis = data.get('tipo')

    if not artista or not cancion:
        return jsonify({"error": "Faltan el artista o el nombre de la canción."})

    resultados = buscar_canciones(f"{artista} {cancion}")
    if not resultados:
        return jsonify({"error": "No se encontraron resultados en Genius."})

    url = resultados[0]["url"]
    imagen = resultados[0]["imagen"]
    letra = obtener_letra_desde_url(url)

    if "Letra no encontrada" in letra or not letra.strip():
        return jsonify({"error": "No se pudo obtener la letra desde Genius."})

    if tipo_analisis == "porcentaje_adjetivos":
        resultado = porcentaje_adjetivos(letra)
    else:
        return jsonify({"error": "Tipo de análisis no soportado."})

    preview_url = obtener_preview_url(artista, cancion)

    return jsonify({
        "letra": letra,
        "resultado": resultado,
        "imagen": imagen,
        "preview_url": preview_url
    })

@app.route('/sugerencias', methods=['GET'])
def sugerencias():
    query = request.args.get('q', '')
    if not query:
        return jsonify([])

    resultados = buscar_canciones(query, max_results=5)
    return jsonify(resultados)

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5001)
