import torch
import os
from analyzer import (
    generate_comprehensive_tabular_report,
    generate_class_examples_image_report,
    generate_all_perturbation_reports
)

# --- IMPORTY LOKALNE ---
from models import MNIST_P1, IMAGENETTE_P1, MLP1
from datasets import get_mnist_datasets, get_imagenette_datasets
from loaders import (
    load_iris_dataset,
    load_wine_dataset,
    load_mnist_projection_feature_dataset
)

# ==========================================
# 0. KONFIGURACJA ŚCIEŻEK MODELI
# ==========================================
MODEL_PATHS = {
    "iris": "models/iris.pth",
    "wine": "models/wine.pth",
    "mnist_projection": "models/mnist_projection_feature.pth",
    "mnist_cnn": "models/model_MNIST_P1_Augment_1_all_data_acc1.00.pth",
    "imagenette_cnn": "models/model_IMAGENETTE_P1_Augment_1_all_data_acc0.7518.pth"
}


# ==========================================
# 1. FUNKCJE ŁADUJĄCE MODELE
# ==========================================
def load_mlp1_model(model_path, device):
    """Ładuje wagi modelu MLP1 z podanej ścieżki."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Nie znaleziono modelu: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    model = MLP1(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_classes=checkpoint["num_classes"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def load_cnn_model(model_class, model_path, device):
    """Ładuje wagi dla klasycznych modeli splotowych (CNN)."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Nie znaleziono modelu: {model_path}")

    model = model_class().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


# ==========================================
# 2. FUNKCJE ANALIZUJĄCE (TABELARYCZNE)
# ==========================================
def run_mlp_tabular_analysis(dataset_name, model, dataset, class_names, device, feature_names=None, include_mean=True):
    """Generuje raporty dla prostych zbiorów tabelarycznych (Iris, Wine)."""
    print(f"\n{'=' * 50}\n--- Analiza MLP1: {dataset_name} ---\n{'=' * 50}")

    # Bezpieczna konwersja do int
    test_data = [(dataset.x[i], dataset.y[i].item() if hasattr(dataset.y[i], 'item') else int(dataset.y[i]))
                 for i in range(len(dataset))]

    generate_comprehensive_tabular_report(
        model=model, dataset=test_data, feature_names=feature_names,
        class_names=class_names, dataset_name=f"{dataset_name}_MLP1", device=device,
        include_mean=include_mean
    )


def run_mnist_projection_analysis(model, dataset, device):
    """Filtruje zbiór MNIST (po 10 przykładów na klasę) i generuje raport."""
    print(f"\n{'=' * 50}\n--- Analiza MLP1: MNIST Projection Feature ---\n{'=' * 50}")

    sampled_data = []
    counts = {i: {'pos': 0, 'neg': 0} for i in range(10)}

    with torch.no_grad():
        for i in range(len(dataset)):
            x = dataset.x[i]
            y = dataset.y[i].item()

            pred = model(x.unsqueeze(0).to(device)).argmax().item()

            if y == pred and counts[y]['pos'] < 10:
                sampled_data.append((x, y))
                counts[y]['pos'] += 1
            elif y != pred and counts[y]['neg'] < 10:
                sampled_data.append((x, y))
                counts[y]['neg'] += 1

            if all(c['pos'] == 10 and c['neg'] == 10 for c in counts.values()):
                break

    print(f"Przetwarzanie próbki: {len(sampled_data)} przykładów.")
    feature_names = [f"Wiersz_{i}" for i in range(28)] + [f"Kol_{i}" for i in range(28)]

    generate_comprehensive_tabular_report(
        model=model, dataset=sampled_data, feature_names=feature_names,
        class_names=[str(i) for i in range(10)], dataset_name="mnist_projection_feature_MLP1",
        device=device, include_mean=False
    )


# ==========================================
# 3. FUNKCJE ANALIZUJĄCE (OBRAZY / CNN)
# ==========================================
def run_cnn_image_analysis(model, dataset_func, dataset_name, device, use_slic=False):
    """Główny orkiestrator analizy dla obrazów (bez ręcznego podawania indeksów)."""
    slic_status = "Z SLIC" if use_slic else "BEZ SLIC"
    print(f"\n{'=' * 50}\n--- Analiza CNN: {dataset_name} | {slic_status} ---\n{'=' * 50}")

    _, _, _, test_dataset, class_names, _ = dataset_func()

    # 1. Szukanie przykładów dla wszystkich klas i generowanie map atrybucji (LIME, FA, IG, Saliency)
    generate_class_examples_image_report(
        model=model, dataset=test_dataset, class_names=class_names,
        dataset_name=f"{dataset_name}_{model.__class__.__name__}",
        device=device, use_slic=use_slic
    )

    # 2. Generowanie wykresów przysłaniania dla każdej klasy
    # Uruchamiamy to tylko dla MNIST, bo w Imagenette ucinanie dołu zdjęcia nie ma większego sensu
    if dataset_name == "MNIST":
        generate_all_perturbation_reports(
            model=model,
            dataset=test_dataset,
            class_names=class_names,
            dataset_name=dataset_name,
            device=device,
            model_name="CNN" # Przekazujemy ładną nazwę do wykresów
        )


# ==========================================
# 4. GŁÓWNA PĘTLA PROGRAMU (STEROWNIK)
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Zainicjowano na urządzeniu: {device}\n")

    # --- USTAWIENIA URUCHOMIENIA ---
    # Zmień na False, aby pominąć obliczenia dla danego zbioru
    RUN_IRIS = False
    RUN_WINE = False
    RUN_MNIST_PROJ = False
    RUN_MNIST_CNN = True
    RUN_IMAGENET = False

    # --- 1. IRIS (MLP1) ---
    if RUN_IRIS:
        try:
            model = load_mlp1_model(MODEL_PATHS["iris"], device)
            _, test_ds = load_iris_dataset()
            run_mlp_tabular_analysis(
                dataset_name="iris", model=model, dataset=test_ds,
                class_names=['Setosa', 'Versicolor', 'Virginica'], device=device,
                feature_names=['sepal length', 'sepal width', 'petal length', 'petal width']
            )
        except Exception as e:
            print(f"Błąd Iris: {e}")

    # --- 2. WINE (MLP1) ---
    if RUN_WINE:
        try:
            model = load_mlp1_model(MODEL_PATHS["wine"], device)
            _, test_ds = load_wine_dataset()
            run_mlp_tabular_analysis(
                dataset_name="wine", model=model, dataset=test_ds,
                class_names=['Class 0', 'Class 1', 'Class 2'], device=device,
                feature_names=['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols',
                               'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue',
                               'OD280/OD315', 'Proline']
            )
        except Exception as e:
            print(f"Błąd Wine: {e}")

    # --- 3. MNIST PROJECTION (MLP1) ---
    if RUN_MNIST_PROJ:
        try:
            model = load_mlp1_model(MODEL_PATHS["mnist_projection"], device)
            _, test_ds = load_mnist_projection_feature_dataset()
            run_mnist_projection_analysis(model, test_ds, device)
        except Exception as e:
            print(f"Błąd MNIST Projection: {e}")

    # --- 4. MNIST OBRAZY (CNN) ---
    if RUN_MNIST_CNN:
        try:
            model = load_cnn_model(MNIST_P1, MODEL_PATHS["mnist_cnn"], device)
            # Automatyczna analiza bez sztywnego indeksu
            run_cnn_image_analysis(model, get_mnist_datasets, "MNIST", device=device, use_slic=False)
        except Exception as e:
            print(f"Błąd MNIST CNN: {e}")

    # --- 5. IMAGENETTE OBRAZY (CNN) ---
    if RUN_IMAGENET:
        try:
            model = load_cnn_model(IMAGENETTE_P1, MODEL_PATHS["imagenette_cnn"], device)
            # Automatyczna analiza bez sztywnego indeksu
            run_cnn_image_analysis(model, get_imagenette_datasets, "Imagenette", device=device, use_slic=False)
        except Exception as e:
            print(f"Błąd IMAGENETTE CNN: {e}")


if __name__ == "__main__":
    main()