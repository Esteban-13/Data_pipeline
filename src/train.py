import os
import mlflow
import mlflow.sklearn
import pandas as pd
import psycopg2
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# 1. Configuration
mlflow.set_experiment("Pipeline_Iris_Classification")

# 2. Chargement des données (Sécurisé pour Docker)
db_host = os.getenv("DB_HOST", "localhost")
db_port = int(os.getenv("DB_PORT", "5432"))
db_name = os.getenv("DB_NAME", "irisdb")
db_user = os.getenv("DB_USER", "iris")
db_password = os.getenv("DB_PASSWORD", "iris")

try:
    print("Tentative de connexion à PostgreSQL...")
    conn = psycopg2.connect(
        host=db_host, port=db_port, dbname=db_name, user=db_user, password=db_password,
        connect_timeout=3
    )
    df = pd.read_sql("SELECT * FROM iris", conn)
    conn.close()
    print("Données chargées depuis SQL.")
except Exception as e:
    print(f"Erreur SQL : {e}")
    print("Repli sur le fichier iris.csv local...")
    df = pd.read_csv("data/iris.csv")

# Nettoyage simple des colonnes si besoin (pour matcher les noms du CSV)
if 'species' not in df.columns and 'target' in df.columns:
    df = df.rename(columns={'target': 'species'})

# Features / cible
feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
X = df[feature_cols]
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Entraînement
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="Classification_Iris"):
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=200)),
        ]
    )
    model.fit(X_train, y_train)

    # Évaluation
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="weighted")

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_weighted", f1)

    print(f"Entraînement terminé avec succès !")
    print(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}")