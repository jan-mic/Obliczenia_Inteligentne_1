import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from model import CNN_Standard_Imagenette, CNN_2Features_Imagenette, CNN_Standard_MNIST, CNN_2Features_MNIST
from dataset import get_mnist_datasets, get_imagenette_datasets, get_balanced_subset
from train import Trainer
from visualize import plot_accuracy_history, plot_confusion_matrix, plot_decision_boundaries, save_augmentation_samples

EXPERIMENT_MODE = 1
DATASET_NAME = "Imagenette"
MODEL_CLASS = CNN_Standard_Imagenette


# ==============================================================================

def run_experiment(dataset_name, current_model_class):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 60}")
    print(f"🚀 ROZPOCZYNAM EKSPERYMENT {EXPERIMENT_MODE} DLA ZBIORU: {dataset_name}")
    print(f"💻 Używane urządzenie obliczeniowe: {device}")
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

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 2. DOSTOSOWANIE PARAMETRÓW DO WYBRANEGO TRYBU
    if EXPERIMENT_MODE == 1:
        data_sizes = [len(train_base)]
        augmentations = {"Brak": train_base}
        num_trials = 1

        print("=> Zapisywanie próbek augmentacji do folderu 'images'...")
        save_augmentation_samples(train_base, train_aug1, "Augmentacja Osoby 1", dataset_name, num_samples=3)
        save_augmentation_samples(train_base, train_aug2, "Augmentacja Osoby 2", dataset_name, num_samples=3)

    elif EXPERIMENT_MODE == 2:
        data_sizes = [100, 200, 1000, len(train_base)]
        augmentations = {
            "Brak": train_base,
            "Augmentacja_Osoba1": train_aug1,
            "Augmentacja_Osoba2": train_aug2
        }
        num_trials = 10
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
                epochs = min(int(base_epochs * (len(full_train_dataset) / size)), 20)
                print(f"Skalowanie epok do: {epochs}")

            train_loader = DataLoader(subset, batch_size=32, shuffle=True)

            accuracy_results = []
            best_overall_acc = 0.0
            best_overall_model = None
            best_train_hist, best_test_hist = [], []

            for i in range(num_trials):
                print(f"  Próba {i + 1}/{num_trials} (Max Epok: {epochs})...")

                # ==============================================================
                # UŻYCIE KLASY TRAINER
                # ==============================================================
                model = current_model_class(num_classes=10).to(device)
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

                # Inicjalizacja naszego nadzorcy
                trainer = Trainer(
                    model=model,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    criterion=criterion,
                    optimizer=optimizer,
                    dataset_name=dataset_name,
                    device=device
                )

                # Uruchomienie uczenia (zabezpieczone przez Early Stopping np. 5 epok)
                history = trainer.fit(epochs=epochs, patience=5)

                # Wyciągamy najwyższe accuracy z historii jako końcowy wynik tej próby
                final_acc = max(history["test_acc"])
                accuracy_results.append(final_acc)
                print(f"  => Zakończono próbę {i + 1}. Najlepsze Acc: {final_acc:.4f}\n")

                if final_acc > best_overall_acc:
                    best_overall_acc = final_acc
                    # Model zapisany w 'trainer.model' ma już załadowane wagi z najlepszej epoki!
                    best_overall_model = trainer.model
                    best_train_hist = history["train_acc"]
                    best_test_hist = history["test_acc"]
                # ==============================================================

            # Podsumowanie z 10 prób (lub 1 próby)
            mean_acc = np.mean(accuracy_results)
            std_acc = np.std(accuracy_results)
            print(f"\n=> WYNIK ({size} danych): Średnia = {mean_acc:.4f}, Odchylenie = {std_acc:.4f}")

            # =================================================================
            # ZAPISYWANIE MODELU I STATYSTYK DO ODPOWIEDNIEGO FOLDERU
            # =================================================================
            save_dir = "models_exp_1" if EXPERIMENT_MODE == 1 else "models_exp_2"
            os.makedirs(save_dir, exist_ok=True)

            model_filename = os.path.join(save_dir,
                                          f"model_{dataset_name}_{current_model_class.__name__}_{aug_name}_size{size}_acc{best_overall_acc:.2f}.pth")
            torch.save(best_overall_model.state_dict(), model_filename)
            print(f"=> Zapisano najlepszy model w: {model_filename}")

            if EXPERIMENT_MODE == 2:
                wyniki_txt_path = os.path.join(save_dir, "wyniki_eksperymentow.txt")
                with open(wyniki_txt_path, "a", encoding="utf-8") as file:
                    file.write(
                        f"Zbiór: {dataset_name} | Model: {current_model_class.__name__} | Augmentacja: {aug_name} | Rozmiar: {size}\n")
                    file.write(f"Średnia Accuracy: {mean_acc:.4f}\n")
                    file.write(f"Odchylenie Standardowe: {std_acc:.4f}\n")
                    file.write("-" * 65 + "\n")
                print(f"=> Zapisano statystyki do pliku: {wyniki_txt_path}")

            # =================================================================
            # 4. WIZUALIZACJA (Tylko w Eksperymencie 1)
            # =================================================================
            if EXPERIMENT_MODE == 1:
                print("=> Generowanie i wyświetlanie wykresów dla tego modelu...")
                plot_accuracy_history(best_train_hist, best_test_hist, title=f"Krzywa uczenia ({dataset_name})")
                plot_confusion_matrix(best_overall_model, test_loader, device, class_names=class_names,
                                      title=f"Macierz_Pomylek_{dataset_name}_{current_model_class.__name__}")

                if current_model_class in [CNN_2Features_MNIST, CNN_2Features_Imagenette]:
                    print("=> Generowanie mapy granic decyzyjnych...")
                    plot_decision_boundaries(best_overall_model, train_loader, device,
                                             title=f"Granice decyzyjne - Trening ({dataset_name})")
                    plot_decision_boundaries(best_overall_model, test_loader, device,
                                             title=f"Granice decyzyjne - Test ({dataset_name})")


if __name__ == '__main__':
    run_experiment(DATASET_NAME, MODEL_CLASS)
