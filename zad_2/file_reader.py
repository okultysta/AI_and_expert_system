import pandas as pd

def read_from_file(filename):
    df = pd.read_excel(filename)

    input_columns = [
        "data__tagData__gyro__x", "data__tagData__gyro__y", "data__tagData__gyro__z",
        "data__orientation__roll", "data__orientation__pitch",
        "data__metrics__latency", "data__metrics__rates__update",
        "data__coordinates__x", "data__coordinates__y", "data__coordinates__z"
    ]

    output_columns = ["reference__x", "reference__y"]

    # Usunięcie wierszy z brakującymi danymi
    df_clean = df.dropna(subset=input_columns + output_columns)

    x_train = df_clean[input_columns].to_numpy()
    y_train = df_clean[output_columns].to_numpy()

    return x_train, y_train



