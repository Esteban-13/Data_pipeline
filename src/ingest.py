import os

import pandas as pd
import psycopg2


def get_db_conn():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        dbname=os.getenv("DB_NAME", "irisdb"),
        user=os.getenv("DB_USER", "iris"),
        password=os.getenv("DB_PASSWORD", "iris"),
    )


def main() -> None:
    data_path = os.getenv("DATA_PATH", "data/iris.csv")
    df = pd.read_csv(data_path)

    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS iris (
                    sepal_length DOUBLE PRECISION,
                    sepal_width DOUBLE PRECISION,
                    petal_length DOUBLE PRECISION,
                    petal_width DOUBLE PRECISION,
                    species TEXT
                );
                """
            )
            cur.execute("TRUNCATE TABLE iris;")

            for _, row in df.iterrows():
                cur.execute(
                    """
                    INSERT INTO iris (sepal_length, sepal_width, petal_length, petal_width, species)
                    VALUES (%s, %s, %s, %s, %s);
                    """,
                    (
                        float(row["sepal_length"]),
                        float(row["sepal_width"]),
                        float(row["petal_length"]),
                        float(row["petal_width"]),
                        row["species"],
                    ),
                )
        conn.commit()

    print("Ingestion terminée.")


if __name__ == "__main__":
    main()
