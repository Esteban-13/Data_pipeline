FROM python:3.11-slim

# Installation des outils pour la base de données PostgreSQL
RUN apt-get update && apt-get install -y libpq-dev gcc && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copie du fichier requirements
COPY requirements.txt .

# Installation des dépendances (celles que tu m'as listées)
RUN pip install --no-cache-dir -r requirements.txt

# Copie de TOUT le contenu du dossier actuel vers le container
# (Cela inclut train.py, iris.csv et le dossier src si tu en as un)
COPY . .

# On expose le port 8000 pour l'API
EXPOSE 8000

# PAR DÉFAUT : On lance l'entraînement
# (Comme ça, quand tu fais "docker run", le modèle se crée)
# On dit à Docker d'aller chercher le fichier DANS le dossier src
CMD ["python", "src/train.py"]