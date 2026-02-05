import os
import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


app = FastAPI(title="Iris Classifier API")

# Variable globale pour le modèle
model = None


def load_model():
    """Charge le dernier modèle depuis MLflow"""
    try:
        # Option 1: Si MODEL_DIR est défini (pour production)
        model_dir = os.getenv("MODEL_DIR")
        if model_dir and os.path.exists(model_dir):
            print(f"📦 Chargement du modèle depuis: {model_dir}")
            return mlflow.sklearn.load_model(model_dir)
        
        # Option 2: Charger depuis MLflow (par défaut)
        mlflow.set_tracking_uri("file:///app/mlruns")
        experiment_name = "Pipeline_Iris_Classification"
        
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise ValueError(f"❌ Expérience '{experiment_name}' non trouvée dans MLflow")
        
        # Récupérer la dernière run
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1
        )
        
        if runs.empty:
            raise ValueError("❌ Aucune run trouvée dans l'expérience")
        
        run_id = runs.iloc[0]["run_id"]
        model_uri = f"runs:/{run_id}/model"
        
        print(f"✅ Chargement du modèle depuis MLflow: {model_uri}")
        return mlflow.sklearn.load_model(model_uri)
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Charge le modèle au démarrage de l'API"""
    global model
    print("🚀 Démarrage de l'API Iris Classifier...")
    try:
        model = load_model()
        print("✅ Modèle chargé avec succès !")
    except Exception as e:
        print(f"⚠️  Impossible de charger le modèle: {e}")


@app.get("/")
def root():
    """Page d'accueil"""
    return {
        "message": "Bienvenue sur l'API de classification Iris",
        "description": "Prédit l'espèce d'une fleur Iris",
        "endpoints": {
            "GET /health": "Vérifier l'état de l'API",
            "POST /predict": "Prédire l'espèce d'une fleur",
            "GET /docs": "Documentation interactive"
        },
        "model_loaded": model is not None
    }


@app.get("/health")
def health():
    """Endpoint de santé"""
    return {
        "status": "ok" if model is not None else "model_not_loaded",
        "model_loaded": model is not None
    }


@app.post("/predict")
def predict(features: IrisFeatures):
    """
    Prédit l'espèce d'une fleur Iris
    
    Args:
        features: Caractéristiques de la fleur (4 mesures)
    
    Returns:
        Espèce prédite (setosa, versicolor, virginica)
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle non chargé. Vérifiez les logs du serveur."
        )
    
    try:
        # Préparer les données
        data = pd.DataFrame(
            [[features.sepal_length, features.sepal_width, 
              features.petal_length, features.petal_width]],
            columns=["sepal_length", "sepal_width", "petal_length", "petal_width"]
        )
        
        # Prédiction
        prediction = model.predict(data)[0]
        
        # Optionnel: obtenir les probabilités
        proba = model.predict_proba(data)[0]
        classes = model.classes_
        
        probabilities = {
            str(cls): float(prob) 
            for cls, prob in zip(classes, proba)
        }
        
        return {
            "species": prediction,
            "probabilities": probabilities
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la prédiction: {str(e)}"
        )


@app.post("/batch_predict")
def batch_predict(features_list: list[IrisFeatures]):
    """
    Prédit l'espèce pour plusieurs fleurs à la fois
    
    Args:
        features_list: Liste de caractéristiques de fleurs
    
    Returns:
        Liste de prédictions
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle non chargé"
        )
    
    try:
        # Préparer les données
        data = pd.DataFrame([
            [f.sepal_length, f.sepal_width, f.petal_length, f.petal_width]
            for f in features_list
        ], columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
        
        # Prédictions
        predictions = model.predict(data)
        
        return {
            "count": len(predictions),
            "predictions": [{"species": pred} for pred in predictions]
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)