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

# 1. Configuration de l'expérience
mlflow.set_experiment("Pipeline_Iris_Classification")

# 2. Chargement des données depuis PostgreSQL
db_host = os.getenv("DB_HOST", "localhost")
db_port = int(os.getenv("DB_PORT", "5432"))
db_name = os.getenv("DB_NAME", "irisdb")
db_user = os.getenv("DB_USER", "iris")
db_password = os.getenv("DB_PASSWORD", "iris")

conn = psycopg2.connect(
    host=db_host, port=db_port, dbname=db_name, user=db_user, password=db_password
)
df = pd.read_sql("SELECT * FROM iris", conn)
conn.close()

# Features / cible
feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
X = df[feature_cols]
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Autologging pour la traçabilité
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="Classification_Iris"):
    # Modèle de classification
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

    # Log manuel des métriques spécifiques
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_weighted", f1)

    # Sauvegarde locale du modèle pour la prédiction (et future dockerisation)
    model_dir = os.getenv("MODEL_DIR", "models/iris_classifier")
    os.makedirs(model_dir, exist_ok=True)
    mlflow.sklearn.save_model(model, model_dir)

    # Log du modèle dans MLflow Tracking (sans Model Registry)
    mlflow.sklearn.log_model(model, artifact_path="iris_classifier_model")

    print(f"Entraînement terminé. Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
