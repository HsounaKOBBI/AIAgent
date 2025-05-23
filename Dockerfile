# Utilise une image Python officielle
FROM python:3.10-slim


# Crée un dossier de travail
WORKDIR /app

# Copie le code
COPY requirements.txt .

# Installe les dépendances
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Téléchargements NLTK
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Port utilisé par Streamlit
EXPOSE 8501

# Commande pour lancer l'app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]