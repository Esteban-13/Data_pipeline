import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Configuration de l'expérience
mlflow.set_experiment("Pipeline_Iris_Regression")

# 2. Chargement des données locales (iris.csv est dans ton dossier)
df = pd.read_csv("iris.csv") 

# Cible : sepal_length | Feature : sepal_width
X = df[['sepal_width']] 
y = df['sepal_length']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Autologging pour la traçabilité [cite: 63]
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="Regression_Sépale"):
    # Modèle de régression
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Évaluation
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    # Log manuel des métriques spécifiques
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2_score", r2)
    
    # Sauvegarde du modèle pour le livrable "Pipeline Dockerisé" 
    mlflow.sklearn.log_model(model, "iris_regression_model")
    
    print(f"Entraînement terminé. MSE: {mse:.4f}, R2: {r2:.4f}")