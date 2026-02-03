import argparse
import os

import mlflow.sklearn
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prédire l'espèce d'iris.")
    parser.add_argument("--sepal_length", type=float, required=True)
    parser.add_argument("--sepal_width", type=float, required=True)
    parser.add_argument("--petal_length", type=float, required=True)
    parser.add_argument("--petal_width", type=float, required=True)
    parser.add_argument(
        "--model_dir",
        type=str,
        default=os.getenv("MODEL_DIR", "models/iris_classifier"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Chargement du modèle depuis un dossier local (Tracking uniquement)
    model = mlflow.sklearn.load_model(args.model_dir)

    # Donnée d'entrée
    data = pd.DataFrame(
        [[args.sepal_length, args.sepal_width, args.petal_length, args.petal_width]],
        columns=["sepal_length", "sepal_width", "petal_length", "petal_width"],
    )

    # Prédiction
    prediction = model.predict(data)
    print(f"Espèce prédite : {prediction[0]}")


if __name__ == "__main__":
    main()
