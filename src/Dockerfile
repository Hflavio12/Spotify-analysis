FROM python

WORKDIR /app

# Copia i file necessari
COPY requirements.txt .
COPY src/ ./src/
COPY models/ ./models/

# Installa le dipendenze
RUN pip install --no-cache-dir -r requirements.txt

# Esponi la porta su cui l'app Flask sarà in ascolto
EXPOSE 5000

# Comando per avviare l'applicazione
CMD ["python", "src/app.py"]