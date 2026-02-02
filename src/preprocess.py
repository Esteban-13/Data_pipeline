# src/preprocess.py
from sklearn.model_selection import train_test_split

def preprocess(df):
    X = df[["sepal_width"]]
    y = df["sepal_length"]

    return train_test_split(
        X, y, test_size=0.2, random_state=42
    )
