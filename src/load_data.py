# src/load_data.py
import pandas as pd

def load_data(path="data/iris.csv"):
    df = pd.read_csv(path)
    return df
