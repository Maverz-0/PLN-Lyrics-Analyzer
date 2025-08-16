from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os

from genius import buscar_canciones, obtener_letra_desde_url

from analyzer import (
    porcentaje_adjetivos, riqueza_lexica, frecuencia_pronombres,
    deteccion_rimas, repeticion_versos, palabras_mas_frecuentes,
    deteccion_temas, deteccion_metaforas, reconocimiento_entidades,
    detectar_idioma, analizar_sentimiento_pro, analizar_sentimiento_simple   # <-- nuevo
)



app = Flask(__name__, static_folder='static')
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

    # 🔤 Detectar idioma (para UI y algunos análisis)
    lang = detectar_idioma(letra)
    lang_code = lang["codigo"]
    lang_conf = lang["confianza"]

    try:
        if tipo_analisis == "porcentaje_adjetivos":
            resultado = porcentaje_adjetivos(letra, lang_code=lang_code)
        elif tipo_analisis == "riqueza_lexica":
            resultado = riqueza_lexica(letra, usar_lemmas=True, excluir_stopwords=True, solo_palabras_contenido=True, lang_code=lang_code)
        elif tipo_analisis == "frecuencia_pronombres":
            resultado = frecuencia_pronombres(letra, incluir_posesivos=True, lang_code=lang_code)
        elif tipo_analisis == "deteccion_rimas":
            resultado = deteccion_rimas(letra, lang_code=lang_code, tail_len=3)
        elif tipo_analisis == "repeticion_versos":
            resultado = repeticion_versos(letra, sim_umbral=0.85, max_versos=250)
        elif tipo_analisis == "palabras_mas_frecuentes":
            resultado = palabras_mas_frecuentes(letra, top_n=10, usar_lemmas=True, excluir_stopwords=True, min_len=2, lang_code=lang_code)
        elif tipo_analisis == "deteccion_temas":
            resultado = deteccion_temas(letra, top_k=3, multi_label=True, lang_code=lang_code)
        elif tipo_analisis == "deteccion_metaforas":
            resultado = deteccion_metaforas(letra, lang_code=lang_code)
        elif tipo_analisis == "reconocimiento_entidades":
            resultado = reconocimiento_entidades(letra, lang_code=lang_code, top_n=8)
        elif tipo_analisis == "sentimiento":
            resultado = analizar_sentimiento_simple(letra, lang_code=lang_code)
        elif tipo_analisis == "sentimiento_pro":
            resultado = analizar_sentimiento_pro(letra)
        else:
            return jsonify({"error": "Tipo de análisis no soportado."})
    except Exception as e:
        # Evita 500 y devuelve JSON que el front entiende
        # Opcional: log detallado
        import traceback; print("ERROR en análisis:", tipo_analisis, e); print(traceback.format_exc())
        return jsonify({"error": f"Error interno en el análisis '{tipo_analisis}': {str(e)}"})

    return jsonify({
        "letra": letra,
        "resultado": resultado,
        "imagen": imagen,
        "idioma": lang_code,
        "confianza_idioma": lang_conf
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


@app.route('/')
def index():
    # Servir index.html desde la carpeta static
    return send_from_directory(os.path.join(app.root_path, 'static'), 'index.html')


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5001)
