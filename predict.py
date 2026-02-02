import mlflow
import pandas as pd

# Récupère l'ID du run depuis l'interface MLflow (ex: "a1b2c3d4...")
run_id = "1d8f070b227740498009e2576a9178dc" 
model_uri = f"runs:/{run_id}/iris_regression_model"

# Chargement du modèle
model = mlflow.pyfunc.load_model(model_uri)

# Test avec une largeur de sépale de 3.5
data = pd.DataFrame([[3.5]], columns=['sepal_width'])
prediction = model.predict(data)

print(f"Pour une largeur de 3.5, la longueur prédite est : {prediction[0]:.2f}")