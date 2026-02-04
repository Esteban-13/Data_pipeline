import argparse
import os
import shutil

import mlflow
import mlflow.sklearn
import pandas as pd
import psycopg2
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entraîner un modèle Iris.")
    parser.add_argument(
        "--model",
        choices=["random_forest", "logreg", "svm", "gradient_boosting", "knn"],
        default="random_forest",
    )
    return parser.parse_args()


args = parse_args()

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

seeds = [1, 2, 3, 4, 5]
metrics = []
best_model = None
best_accuracy = -1.0

for seed in seeds:
    with mlflow.start_run(run_name=f"Classification_Iris_seed_{seed}"):
        # Modèle de classification
        if args.model == "random_forest":
            model = RandomForestClassifier(
                n_estimators=200,
                random_state=seed,
            )
        elif args.model == "svm":
            model = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("clf", SVC(kernel="rbf", C=1.0, gamma="scale")),
                ]
            )
        elif args.model == "gradient_boosting":
            model = GradientBoostingClassifier(random_state=seed)
        elif args.model == "knn":
            model = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("clf", KNeighborsClassifier(n_neighbors=5)),
                ]
            )
        else:
            model = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=200, random_state=seed)),
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

        metrics.append((accuracy, f1))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

# Sauvegarde locale du meilleur modèle pour la prédiction (et future dockerisation)
model_dir = os.getenv("MODEL_DIR", "models/iris_classifier")
if os.path.isdir(model_dir):
    shutil.rmtree(model_dir)
os.makedirs(model_dir, exist_ok=True)
mlflow.sklearn.save_model(best_model, model_dir)

# Run de synthèse pour les moyennes
avg_accuracy = sum(m[0] for m in metrics) / len(metrics)
avg_f1 = sum(m[1] for m in metrics) / len(metrics)

with mlflow.start_run(run_name="Classification_Iris_Summary"):
    mlflow.log_metric("avg_accuracy", avg_accuracy)
    mlflow.log_metric("avg_f1_weighted", avg_f1)
    mlflow.sklearn.log_model(best_model, artifact_path="iris_classifier_model")

print(f"Entraînements terminés. Moyenne Accuracy: {avg_accuracy:.4f}, F1: {avg_f1:.4f}")
