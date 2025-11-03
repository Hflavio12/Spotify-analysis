# CORREZIONE: Usiamo una versione di Python compatibile con TensorFlow 2.16.x
FROM python:3.10-slim

# Installa gli strumenti di build necessari per librerie come Matplotlib e TensorFlow
RUN apt-get update && \ 
    # CORREZIONE: Sostituito libatlas-base-dev (non trovato) con libopenblas-dev,
    # che fornisce le librerie di algebra lineare necessarie per Numpy/TensorFlow.
    apt-get install -y build-essential libopenblas-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copia i requisiti e installa le dipendenze Python
COPY src/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia le cartelle necessarie
# 1. Copia la cartella dati (necessaria per data_loader.py/main.py)
COPY data/ ./data/
# 2. Copia tutto il codice sorgente (che include app.py, model.py, e spotify_model.h5)
COPY src/ ./src/

# Espongo la porta 5000 (standard per Flask)
EXPOSE 5000

# Comando per avviare l'applicazione API
# Nota: Esecuzione dalla directory /app
CMD ["python", "src/app.py"]
