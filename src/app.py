from flask import Flask, request, jsonify
from tensorflow import keras
import numpy as np
import os

app = Flask(__name__)

# CORREZIONE CRITICA: Uso del percorso relativo all'interno del container.
# Il modello .h5 è in src/, quindi il percorso è relativo al punto di esecuzione del comando CMD.
MODEL_PATH = "src/spotify_model.h5" 
# Alternativamente, se il modello fosse in una cartella 'models', sarebbe 'models/spotify_model.h5'

try:
    model = keras.models.load_model(MODEL_PATH)
    print(f"✓ Modello Keras caricato da: {MODEL_PATH}")
except Exception as e:
    print(f"ERRORE CRITICO nel caricamento del modello: {e}")
    # Uscita o gestione errore se il modello non può essere caricato
    model = None 

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
        
    try:
        data = request.get_json()
        # Assumendo che 'features' sia una lista di 1D array per la previsione
        features = np.array([data["features"]])  
        
        prediction = model.predict(features)
        
        return jsonify({"predicted_popularity": float(prediction[0][0])})
    except Exception as e:
        return jsonify({"error": str(e), "message": "Check the input feature format."}), 400

if __name__ == "__main__":
    # Usa la porta 5000 come precedentemente esposto nel Dockerfile
    app.run(host="0.0.0.0", port=5000)