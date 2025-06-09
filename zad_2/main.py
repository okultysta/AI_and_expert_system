# Re-importy i re-definicje po resecie środowiska

import numpy as np
import file_reader
import matplotlib.pyplot as plt
import network_file
import pandas as pd
from sklearn.preprocessing import StandardScaler

def save_and_plot_cdf_before_filtering(measured, real):
    print("\n--- DYSTRYBUANTA PRZED FILTRACJĄ ---")

    # Oblicz błąd euklidesowy dla każdej próbki (bez filtracji)
    diffs = measured - real
    errors = np.linalg.norm(diffs, axis=1)

    # Posortuj błędy i oblicz dystrybuantę
    sorted_errors = np.sort(errors)
    cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)

    # Zapis do pliku Excel
    df = pd.DataFrame({'Dystrybuanta_przed_filtracja': cdf})
    df.to_excel("dystrybuanta_przed_filtracja.xlsx", index=False)
    print("Zapisano plik 'dystrybuanta_przed_filtracja.xlsx'.")

    # Wykres
    plt.figure(figsize=(8, 5))
    plt.plot(sorted_errors, cdf, label='Dystrybuanta błędu (przed filtracją)', color='red')
    plt.xlabel("Błąd euklidesowy")
    plt.ylabel("Dystrybuanta")
    plt.title("Dystrybuanta błędu pomiarowego przed filtracją")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("cdf_przed_filtracja.png")
    print("Zapisano wykres 'cdf_przed_filtracja.png'.")
    plt.close()

def plot_xy_comparison(measured, predicted, reference, filename="trajektoria_test_random.png"):
    plt.figure(figsize=(12, 7))

    plt.plot(reference[:, 0], reference[:, 1], label="Rzeczywista", color='gold', linewidth=2, zorder=1)
    plt.plot(measured[:, 0], measured[:, 1], 'r.', label="Zmierzona (UWB)", markersize=4, zorder=2)
    plt.plot(predicted[:, 0], predicted[:, 1], 'b.', label="Po korekcji (sieć)", markersize=4, zorder=3)

    plt.xlabel("X [mm]")
    plt.ylabel("Y [mm]")
    plt.title("Porównanie trajektorii: zmierzona vs poprawiona vs rzeczywista")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Wykres trajektorii zapisany jako '{filename}'")
    plt.close()


def test_model(network, input_scaler, output_scaler, test_type="random"):
    print(f"\n--- TESTOWANIE MODELU NA DANYCH: '{test_type.upper()}' ---")

    if test_type == "random":
        file_type = 1
        prefix = "random"
    elif test_type == "dynamic":
        file_type = 3
        prefix = "dynamic"
    else:
        print(f"Błąd: nieznany typ testu '{test_type}'. Użyj 'random' lub 'dynamic'.")
        return

    measured_test, real_test = file_reader.read_all_static_files_from_directory("data", file_type)

    if measured_test is None or real_test is None:
        print(f"Brak danych testowych typu '{test_type}'.")
        return

    # Normalizacja
    measured_test_norm = input_scaler.transform(measured_test)

    # Przewidywanie i denormalizacja
    predictions_norm = np.array([network.forward(x) for x in measured_test_norm])
    predictions = output_scaler.inverse_transform(predictions_norm)

    # Błędy: przed i po filtracji
    errors_before = np.linalg.norm(measured_test - real_test, axis=1)
    errors_after = np.linalg.norm(predictions - real_test, axis=1)

    print(f"\n📊 Statystyki błędu ({test_type}):")
    print(f"Średni błąd PRZED filtracją: {np.mean(errors_before):.4f}")
    print(f"Średni błąd PO filtracji:    {np.mean(errors_after):.4f}")
    print(f"95 percentyl PRZED: {np.percentile(errors_before, 95):.4f}")
    print(f"95 percentyl PO:    {np.percentile(errors_after, 95):.4f}")

    # Dystrybuanty
    sorted_before = np.sort(errors_before)
    sorted_after = np.sort(errors_after)
    cdf = np.arange(1, len(sorted_before) + 1) / len(sorted_before)

    # Zapis do pliku
    df = pd.DataFrame({
        "Błąd_przed": sorted_before,
        "CDF_przed": cdf,
        "Błąd_po": sorted_after,
        "CDF_po": cdf
    })
    df.to_excel(f"dystrybuanta_{prefix}.xlsx", index=False)
    print(f"📁 Zapisano dystrybuanty do pliku 'dystrybuanta_{prefix}.xlsx'")

    # Wykres dystrybuanty
    plt.figure(figsize=(8, 5))
    plt.plot(sorted_before, cdf, 'r--', label='Przed filtracją')
    plt.plot(sorted_after, cdf, 'b-', label='Po filtracji')
    plt.xlabel("Błąd euklidesowy [mm]")
    plt.ylabel("Dystrybuanta (CDF)")
    plt.title(f"Porównanie dystrybuanty – {prefix}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"cdf_{prefix}.png")
    print(f"✅ Wykres zapisany jako 'cdf_{prefix}.png'")
    plt.close()

    # Wykres trajektorii
    plot_xy_comparison(measured_test, predictions, real_test, filename=f"trajektoria_test_{prefix}.png")



training = input("Czy chcesz trenować model?[Y/N]:")


network = network_file.SimpleNeuralNetwork(input_dim=2)
network.load("model_weights.pkl")  # załaduj model jeśli istnieje

measured, real = file_reader.read_all_static_files_from_directory("data", 0)

input_scaler = StandardScaler()
output_scaler = StandardScaler()

measured_norm = input_scaler.fit_transform(measured)
real_norm = output_scaler.fit_transform(real)

if training == "Y" or training == "y":
    network.train(measured_norm, real_norm, epochs=2, learning_rate=0.001)
    network.save("model_weights.pkl")  # końcowy zapis wag
measured, real = file_reader.read_all_static_files_from_directory("data", 1)
save_and_plot_cdf_before_filtering(measured, real)
test_model(network, input_scaler, output_scaler, test_type="random")
test_model(network, input_scaler, output_scaler, test_type="dynamic")

