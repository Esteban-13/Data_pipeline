# src/train.py
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression

def train_model(X_train, y_train):
    with mlflow.start_run():
        model = LinearRegression()
        model.fit(X_train, y_train)

        mlflow.log_param("model", "LinearRegression")
        mlflow.log_metric("train_score", model.score(X_train, y_train))
        mlflow.sklearn.log_model(model, "model")

        return model
