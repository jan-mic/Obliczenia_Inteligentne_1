import numpy as np
from sklearn.datasets import load_iris, load_wine


def print_dataset_stats_numpy(dataset_loader, name):
    dataset = dataset_loader()
    X = dataset.data
    features = dataset.feature_names

    print(f" Statystyki dla zbioru: {name.upper()}")

    print(f"{'Cecha (Feature)':<30} | {'Średnia':>8} | {'Odchyl.':>8} | {'Min':>6} | {'Max':>8} | {'Mediana':>8}")
    print("-" * 75)


    for i, feature_name in enumerate(features):
        col_data = X[:, i]

        mean_val = np.mean(col_data)
        std_val = np.std(col_data)
        min_val = np.min(col_data)
        max_val = np.max(col_data)
        median_val = np.median(col_data)


        print(
            f"{feature_name[:28]:<30} | {mean_val:8.3f} | {std_val:8.3f} | {min_val:6.3f} | {max_val:8.3f} | {median_val:8.3f}")



if __name__ == "__main__":
    print_dataset_stats_numpy(load_iris, 'Iris')
    print_dataset_stats_numpy(load_wine, 'Wine')