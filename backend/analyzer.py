import spacy

# Usamos modelo en español
nlp = spacy.load("es_core_news_sm")

def porcentaje_adjetivos(texto):
    doc = nlp(texto)
    total_palabras = len([token for token in doc if token.is_alpha])
    adjetivos = len([token for token in doc if token.pos_ == "ADJ"])

    if total_palabras == 0:
        return {"porcentaje": 0, "adjetivos": 0, "total": 0}

    porcentaje = round((adjetivos / total_palabras) * 100, 2)
    return {
        "porcentaje": porcentaje,
        "adjetivos": adjetivos,
        "total": total_palabras
    }

d
