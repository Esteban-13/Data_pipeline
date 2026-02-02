# src/evaluate.py
from sklearn.metrics import mean_squared_error
import mlflow

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mlflow.log_metric("mse", mse)
    return mse
