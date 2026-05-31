import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

# Importy wszystkich modeli
from model import MNIST_P1, MNIST_P1_2F, MNIST_P2, MNIST_P2_2F, IMAGENETTE_P1, IMAGENETTE_P1_2F, IMAGENETTE_P2, \
    IMAGENETTE_P2_2F
from dataset import get_mnist_datasets, get_imagenette_datasets, get_balanced_subset
from train import Trainer
from visualize import plot_accuracy_history, plot_confusion_matrix, plot_decision_boundaries, save_augmentation_samples

# ==============================================================================
# TRYB EKSPERYMENTU
# 1 - Pojedyncza próba, pełny zbiór, generowanie wykresów (Zapis do eksperyment_1)
# 2 - Wiele prób, zmiana rozmiarów zbiorów, zapis statystyk (Zapis do eksperyment_2)
EXPERIMENT_MODE = 2


# ==============================================================================

def run_experiment(dataset_name, current_model_class, save_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 60}")
    print(
        f"🚀 ROZPOCZYNAM EKSPERYMENT {EXPERIMENT_MODE} DLA ZBIORU: {dataset_name} | MODEL: {current_model_class.__name__}")
    print(f"💻 Używane urządzenie obliczeniowe: {device}")
    print(f"📁 Wyniki zostaną zapisane w: {save_dir}")
    print(f"{'=' * 60}\n")

    # 1. POBIERANIE DANYCH
    if dataset_name == "MNIST":
        train_base, train_aug1, train_aug2, test_dataset, class_names, input_channels = get_mnist_datasets()
        base_epochs = 15
    elif dataset_name == "Imagenette":
        train_base, train_aug1, train_aug2, test_dataset, class_names, input_channels = get_imagenette_datasets()
        base_epochs = 20
    else:
        raise ValueError("Nieobsługiwany zbiór danych.")

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    # 2. DOSTOSOWANIE PARAMETRÓW DO WYBRANEGO TRYBU
    if EXPERIMENT_MODE == 1:
        data_sizes = [len(train_base)]
        augmentations = {"Brak": train_base}
        num_trials = 1

        print("=> Zapisywanie próbek augmentacji...")
        save_augmentation_samples(train_base, train_aug1, "Augment_1", dataset_name, num_samples=3, save_dir=save_dir)
        save_augmentation_samples(train_base, train_aug2, "Augment_2", dataset_name, num_samples=3, save_dir=save_dir)

    elif EXPERIMENT_MODE == 2:
        data_sizes = [100, 200, 1000, len(train_base)]
        augmentations = {
            "Brak": train_base,
            "Augment_1": train_aug1,
            "Augment_2": train_aug2
        }
        num_trials = 5
    else:
        raise ValueError("Nierozpoznany tryb eksperymentu (wybierz 1 lub 2).")

    # 3. GŁÓWNA PĘTLA EKSPERYMENTÓW
    for aug_name, full_train_dataset in augmentations.items():
        print(f"\n{'*' * 60}\n** TESTY DLA AUGMENTACJI: {aug_name.upper()} **\n{'*' * 60}")

        for size in data_sizes:
            print(f"\n--- Rozmiar zbioru treningowego: {size} ---")

            if size == len(full_train_dataset):
                subset = full_train_dataset
                epochs = base_epochs
            else:
                subset = get_balanced_subset(full_train_dataset, size)
                epochs = min(int(base_epochs * (len(full_train_dataset) / size)), 30)
                print(f"Skalowanie epok do: {epochs}")

            train_loader = DataLoader(
                subset,
                batch_size=32,
                shuffle=True,
                num_workers=8,
                pin_memory=True
            )

            accuracy_results = []
            best_overall_acc = 0.0
            best_overall_model = None
            best_train_hist, best_test_hist = [], []

            for i in range(num_trials):
                print(f"  Próba {i + 1}/{num_trials} (Max Epok: {epochs})...")

                model = current_model_class(num_classes=10).to(device)
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

                trainer = Trainer(
                    model=model,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    criterion=criterion,
                    optimizer=optimizer,
                    dataset_name=dataset_name,
                    device=device
                )

                history = trainer.fit(epochs=epochs, patience=5)

                final_acc = max(history["test_acc"])
                accuracy_results.append(final_acc)
                print(f"  => Zakończono próbę {i + 1}. Najlepsze Acc: {final_acc:.4f}\n")

                if final_acc > best_overall_acc:
                    best_overall_acc = final_acc
                    best_overall_model = trainer.model
                    best_train_hist = history["train_acc"]
                    best_test_hist = history["test_acc"]

            mean_acc = np.mean(accuracy_results)
            std_acc = np.std(accuracy_results)
            print(f"\n=> WYNIK ({size} danych): Średnia = {mean_acc:.4f}, Odchylenie = {std_acc:.4f}")

            # =================================================================
            # ⚙️ LOGIKA NAZEWNICTWA PLIKÓW (all_data lub wielkość zbioru)
            # =================================================================
            size_str = "all_data" if size == len(full_train_dataset) else f"size_{size}"

            # Zapisywanie wag modelu (.pth)
            models_dir = os.path.join(save_dir, "models")
            os.makedirs(models_dir, exist_ok=True)

            model_filename = os.path.join(models_dir,
                                          f"model_{current_model_class.__name__}_{aug_name}_{size_str}_acc{best_overall_acc:.2f}.pth")
            torch.save(best_overall_model.state_dict(), model_filename)
            print(f"=> Zapisano najlepszy model w: {model_filename}")

            # Zapisywanie statystyk tekstowych (.txt) - tylko dla Eksperymentu 2
            if EXPERIMENT_MODE == 2:
                wyniki_txt_path = os.path.join(save_dir, f"wyniki_eksperymentow_{size_str}.txt")
                with open(wyniki_txt_path, "a", encoding="utf-8") as file:
                    file.write(
                        f"Model: {current_model_class.__name__} | Augmentacja: {aug_name} | Rozmiar: {size_str}\n")
                    file.write(f"Średnia Accuracy: {mean_acc:.4f}\n")
                    file.write(f"Odchylenie Standardowe: {std_acc:.4f}\n")
                    file.write("-" * 65 + "\n")
                print(f"=> Zapisano statystyki do pliku: {wyniki_txt_path}")

            # =================================================================
            # 4. WIZUALIZACJA
            # =================================================================
            if EXPERIMENT_MODE == 1:
                print("=> Generowanie i wyświetlanie wykresów dla tego modelu...")
                plot_accuracy_history(best_train_hist, best_test_hist, title=f"Accuracy ({dataset_name})",
                                      save_dir=save_dir)
                plot_confusion_matrix(best_overall_model, test_loader, device, class_names=class_names,
                                      title=f"confusion_matrix_{current_model_class.__name__}",
                                      save_dir=save_dir)

                if current_model_class in [MNIST_P1_2F, MNIST_P2_2F, IMAGENETTE_P1_2F, IMAGENETTE_P2_2F]:
                    print("=> Generowanie mapy granic decyzyjnych...")
                    plot_decision_boundaries(best_overall_model, train_loader, device,
                                             title=f"Granice_decyzyjne_Trening_{dataset_name}", save_dir=save_dir)
                    plot_decision_boundaries(best_overall_model, test_loader, device,
                                             title=f"Granice_decyzyjne_Test_{dataset_name}", save_dir=save_dir)


if __name__ == '__main__':
    # Ustalamy dynamicznie podfolder na podstawie aktywnego trybu eksperymentu
    EXPERIMENT_MODE = 2
    path = f"eksperyment_{EXPERIMENT_MODE}"

    # # --- 1. PĘTLA DLA ZBIORU MNIST ---
    # mnist_models = [MNIST_P1, MNIST_P1_2F, MNIST_P2, MNIST_P2_2F]
    # print(f"=== Uruchamiam sekwencję dla {len(mnist_models)} modeli MNIST (Tryb: {EXPERIMENT_MODE}) ===")
    #
    # for _, model in enumerate(mnist_models, start=1):
    #     run_experiment(
    #         dataset_name="MNIST",
    #         current_model_class=model,
    #         save_dir=f"{path}/{model.__name__}"
    #     )
    #
    # # --- 2. PĘTLA DLA ZBIORU IMAGENETTE ---
    # imagenette_models = [IMAGENETTE_P1, IMAGENETTE_P1_2F, IMAGENETTE_P2, IMAGENETTE_P2_2F]
    # print(f"\n=== Uruchamiam sekwencję dla {len(imagenette_models)} modeli Imagenette (Tryb: {EXPERIMENT_MODE}) ===")
    #
    # for _, model in enumerate(imagenette_models, start=1):
    #     run_experiment(
    #         dataset_name="Imagenette",
    #         current_model_class=model,
    #         save_dir=f"{path}/{model.__name__}"
    #     )

    ################################
    # mnist_models = [MNIST_P1, MNIST_P1_2F]
    # print(f"=== Uruchamiam sekwencję dla {len(mnist_models)} modeli MNIST (Tryb: {EXPERIMENT_MODE}) ===")
    #
    # for _, model in enumerate(mnist_models, start=1):
    #     run_experiment(
    #         dataset_name="MNIST",
    #         current_model_class=model,
    #         save_dir=f"{path}/{model.__name__}"
    #     )

    # --- 2. PĘTLA DLA ZBIORU IMAGENETTE ---
    # Zostawiamy TYLKO najlepszy model
    imagenette_models = [IMAGENETTE_P2, IMAGENETTE_P1_2F]
    print(f"\n=== Uruchamiam sekwencję dla {len(imagenette_models)} modeli Imagenette (Tryb: {EXPERIMENT_MODE}) ===")

    for _, model in enumerate(imagenette_models, start=1):
        run_experiment(
            dataset_name="Imagenette",
            current_model_class=model,
            save_dir=f"{path}/{model.__name__}"
        )