# Re-importy i re-definicje po resecie środowiska

import numpy as np
import file_reader
import matplotlib.pyplot as plt
import network_file


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


def test_model_on_random(network, measured_mean, measured_std, real_mean, real_std):
    print("\n--- TESTOWANIE MODELU NA DANYCH 'RANDOM' ---")
    measured_test, real_test = file_reader.read_all_static_files_from_directory("data", 1)

    if measured_test is None or real_test is None:
        print("Brak danych testowych typu 'random'.")
        return

    # Normalizacja danymi ze zbioru treningowego
    measured_test_norm = (measured_test - measured_mean) / measured_std

    # Przewidywanie
    predictions_norm = np.array([network.forward(x) for x in measured_test_norm])
    predictions = predictions_norm * real_std + real_mean  # denormalizacja

    # Błąd euklidesowy
    diffs = predictions - real_test
    errors = np.linalg.norm(diffs, axis=1)

    print("\nStatystyki błędu na zbiorze 'random':")
    print(f"Średni błąd: {np.mean(errors):.4f}")
    print(f"Mediana błędu: {np.median(errors):.4f}")
    print(f"95 percentyl błędu: {np.percentile(errors, 95):.4f}")
    # Rysowanie porównania
    plot_xy_comparison(measured_test, predictions, real_test, filename="trajektoria_test_random.png")




network = network_file.SimpleNeuralNetwork(input_dim=2)
network.load("model_weights.pkl")  # załaduj model jeśli istnieje

measured, real = file_reader.read_all_static_files_from_directory("data", 0)

measured_mean = np.mean(measured, axis=0)
measured_std = np.std(measured, axis=0)
measured_std[measured_std == 0] = 1  # unikanie dzielenia przez 0

real_mean = np.mean(real, axis=0)
real_std = np.std(real, axis=0)
real_std[real_std == 0] = 1

measured_norm = (measured - measured_mean) / measured_std
real_norm = (real - real_mean) / real_std

network.train(measured_norm, real_norm, epochs=50, learning_rate=0.001)
network.save("model_weights.pkl")  # końcowy zapis wag
test_model_on_random(network, measured_mean, measured_std, real_mean, real_std)