from flask import Flask, request, jsonify
from tensorflow import keras
import numpy as np

app = Flask(__name__)
model = keras.models.load_model("C:\\Users\\Daniele\\Desktop\\Spotify_project\\Spotify\\src\\spotify_model.h5")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = np.array([data["features"]])  # es. [0.5, 0.8, 0.1, ...]
    prediction = model.predict(features)
    return jsonify({"predicted_popularity": float(prediction[0][0])})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)