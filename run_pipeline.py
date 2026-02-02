# run_pipeline.py
from src.load_data import load_data
from src.preprocess import preprocess
from src.train import train_model
from src.evaluate import evaluate_model

df = load_data()
X_train, X_test, y_train, y_test = preprocess(df)
model = train_model(X_train, y_train)
mse = evaluate_model(model, X_test, y_test)

print(f"MSE: {mse}")
