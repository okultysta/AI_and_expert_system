import os
import glob
import pandas as pd
import numpy as np

# używane kolumny tylko dla danych typu "random"
input_columns_random = ["data__coordinates__x", "data__coordinates__y"]
output_columns = ["reference__x", "reference__y"]

def read_from_file(filename):
    df = pd.read_excel(filename)

    df_clean = df.dropna(subset=input_columns_random + output_columns)

    x_train = df_clean[input_columns_random].to_numpy()
    y_train = df_clean[output_columns].to_numpy()

    return x_train, y_train


def read_all_static_files_from_directory(directory, file_type):
    if file_type == 0:
        pattern = os.path.join(directory, '*stat*.xlsx')
        file_list = glob.glob(pattern)
    elif file_type == 1:
        pattern = os.path.join(directory, '*random*.xlsx')
        file_list = glob.glob(pattern)
    elif file_type == 3:
        pattern = os.path.join(directory, '*.xlsx')
        all_files = glob.glob(pattern)
        file_list = [f for f in all_files if 'stat' not in f and 'random' not in f]
    else:
        pattern = os.path.join(directory, '*.xlsx')
        file_list = glob.glob(pattern)

    if not file_list:
        print(f"Nie znaleziono plików w folderze: {directory}")
        return None, None

    x_list = []
    y_list = []

    for file in file_list:
        try:
            df = pd.read_excel(file)
            df_clean = df.dropna(subset=input_columns_random + output_columns)

            x = df_clean[input_columns_random].to_numpy()
            y = df_clean[output_columns].to_numpy()

            x_list.append(x)
            y_list.append(y)

            print(f"Wczytano: {os.path.basename(file)} ({len(df_clean)} próbek)")
        except Exception as e:
            print(f"Błąd podczas wczytywania pliku {file}: {e}")

    if not x_list:
        print("Żaden plik nie zawierał poprawnych danych.")
        return None, None

    x_all = np.vstack(x_list)
    y_all = np.vstack(y_list)

    print(f"\nŁącznie załadowano: {x_all.shape[0]} próbek z {len(file_list)} plików.")
    return x_all, y_all
