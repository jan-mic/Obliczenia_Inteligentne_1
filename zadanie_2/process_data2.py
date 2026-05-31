import numpy as np
from sklearn.datasets import load_iris, load_wine
from sklearn.preprocessing import StandardScaler


def print_scaled_stats_and_example(dataset_loader, name):
    # 1. Ładujemy zbiór danych
    dataset = dataset_loader()
    X = dataset.data
    features = dataset.feature_names
    target_names = dataset.target_names  # Pobieramy nazwy klas docelowych

    # 2. Inicjalizujemy i używamy StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"\n====================================================================================")
    print(f" Zbiór: {name.upper()} - PO STANDARYZACJI (StandardScaler)")
    print(f"====================================================================================")

    # 3. Wyświetlamy przykład znormalizowanych danych (pierwsze 3 wiersze)
    print("--- Przykład przetworzonych danych (pierwsze 3 próbki) ---")
    for i in range(3):
        row_values = X_scaled[i, :5]
        row_str = " | ".join([f"{val:7.3f}" for val in row_values])

        if X_scaled.shape[1] > 5:
            row_str += " | ..."

        print(f" Próbka {i + 1}: [ {row_str} ]")
    print("-" * 84)

    # 4. Drukujemy statystyki DLA ZNORMALIZOWANYCH DANYCH
    print(f"{'Cecha (Feature)':<30} | {'Średnia':>8} | {'Odchyl.':>8} | {'Min':>8} | {'Max':>8} | {'Mediana':>8}")
    print("-" * 84)

    for i, feature_name in enumerate(features):
        col_data = X_scaled[:, i]

        mean_val = np.mean(col_data)
        std_val = np.std(col_data)
        min_val = np.min(col_data)
        max_val = np.max(col_data)
        median_val = np.median(col_data)

        print(
            f"{feature_name[:28]:<30} | {mean_val:8.3f} | {std_val:8.3f} | {min_val:8.3f} | {max_val:8.3f} | {median_val:8.3f}")

    # 5. Wyświetlamy dostępne etykiety klas
    print("-" * 84)
    # Tworzymy ładny napis mapujący numer na nazwę, np. "0: setosa, 1: versicolor"
    labels_mapped = ", ".join([f"{i}: {label_name}" for i, label_name in enumerate(target_names)])
    print(f" Dostępne klasy (etykiety): {labels_mapped}")
    print("====================================================================================\n")


if __name__ == "__main__":
    print_scaled_stats_and_example(load_iris, 'Iris')
    print_scaled_stats_and_example(load_wine, 'Wine')