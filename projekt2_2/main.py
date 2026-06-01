import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from model import MNIST_P1, MNIST_P1_2F, MNIST_P2, MNIST_P2_2F, IMAGENETTE_P1, IMAGENETTE_P1_2F, IMAGENETTE_P2, \
    IMAGENETTE_P2_2F
from dataset import get_mnist_datasets, get_imagenette_datasets, get_balanced_subset
from train import Trainer
from visualize import plot_accuracy_history, plot_confusion_matrix, plot_decision_boundaries, save_augmentation_samples


def run_experiment(dataset_name, current_model_class, save_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"EKSPERYMENT {EXPERIMENT_MODE} | ZBIOR: {dataset_name} | MODEL: {current_model_class.__name__}")

    if dataset_name == "MNIST":
        train_base, train_aug1, train_aug2, test_dataset, class_names, input_channels = get_mnist_datasets()
        base_epochs = 15
    elif dataset_name == "Imagenette":
        train_base, train_aug1, train_aug2, test_dataset, class_names, input_channels = get_imagenette_datasets()
        base_epochs = 15

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    if EXPERIMENT_MODE == 1:
        data_sizes = [len(train_base)]
        augmentations = {"Brak": train_base}
        num_training = 1

        save_augmentation_samples(train_base, train_aug1, "Augment_1", dataset_name, num_samples=3, save_dir=save_dir)
        save_augmentation_samples(train_base, train_aug2, "Augment_2", dataset_name, num_samples=3, save_dir=save_dir)

    elif EXPERIMENT_MODE == 2:
        data_sizes = [100, 200, 1000, len(train_base)]
        augmentations = {
            "Brak": train_base,
            "Augment_1": train_aug1,
            "Augment_2": train_aug2
        }
        num_training = 10

    for aug_name, full_train_dataset in augmentations.items():
        print(f"\nAugmentacja: {aug_name}")

        for size in data_sizes:
            print(f"\nRozmiar zbioru: {size}")

            if size == len(full_train_dataset):
                subset = full_train_dataset
                epochs = base_epochs
            else:
                subset = get_balanced_subset(full_train_dataset, size)
                epochs = min(int(base_epochs * (len(full_train_dataset) / size)), 30)

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

            for i in range(num_training):
                print(f"Trening {i + 1}/{num_training}")

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
                print(f"Trening {i + 1}. Best Acc: {final_acc:.4f}\n")

                if final_acc > best_overall_acc:
                    best_overall_acc = final_acc
                    best_overall_model = trainer.model
                    best_train_hist = history["train_acc"]
                    best_test_hist = history["test_acc"]

            mean_acc = np.mean(accuracy_results)
            std_acc = np.std(accuracy_results)
            print(f"\nRozmiar danych: {size} | Srednia = {mean_acc:.4f} | Odchylenie = {std_acc:.4f}")

            size_name = "all_data" if size == len(full_train_dataset) else f"size_{size}"
            models_dir = os.path.join(save_dir, "models")
            os.makedirs(models_dir, exist_ok=True)
            model_filename = os.path.join(models_dir,
                                          f"model_{current_model_class.__name__}_{aug_name}_{size_name}_acc{best_overall_acc:.4f}.pth")
            torch.save(best_overall_model.state_dict(), model_filename)

            if EXPERIMENT_MODE == 2:
                wyniki_txt_path = os.path.join(save_dir, f"wyniki_eksperymentow_{size_name}.txt")
                with open(wyniki_txt_path, "a", encoding="utf-8") as file:
                    file.write(
                        f"Model: {current_model_class.__name__} | Augmentacja: {aug_name} | Rozmiar: {size_name}\n")
                    file.write(f"Srednia Accuracy: {mean_acc:.4f}\n")
                    file.write(f"Odchylenie Standardowe: {std_acc:.4f}\n")

            if EXPERIMENT_MODE == 1:
                plot_accuracy_history(best_train_hist, best_test_hist, dataset_name=f"Accuracy ({dataset_name})",
                                      save_dir=save_dir)
                plot_confusion_matrix(best_overall_model, test_loader, device, class_names=class_names,
                                      dataset_name_cm=f"confusion_matrix_{current_model_class.__name__}",
                                      save_dir=save_dir)

                if current_model_class in [MNIST_P1_2F, MNIST_P2_2F, IMAGENETTE_P1_2F, IMAGENETTE_P2_2F]:
                    plot_decision_boundaries(best_overall_model, train_loader, device,
                                             dataset_name=f"Granice_decyzyjne_Trening_{dataset_name}",
                                             save_dir=save_dir)
                    plot_decision_boundaries(best_overall_model, test_loader, device,
                                             dataset_name=f"Granice_decyzyjne_Test_{dataset_name}", save_dir=save_dir)


if __name__ == '__main__':
    global EXPERIMENT_MODE
    EXPERIMENT_MODE = 1
    file_path = f"eksperyment_{EXPERIMENT_MODE}"

    mnist_models = [MNIST_P1, MNIST_P1_2F, MNIST_P2, MNIST_P2_2F]
    for model in mnist_models:
        run_experiment(
            dataset_name="MNIST",
            current_model_class=model,
            save_dir=f"{file_path}/{model.__name__}"
        )

    imagenette_models = [IMAGENETTE_P1, IMAGENETTE_P1_2F, IMAGENETTE_P2, IMAGENETTE_P2_2F]
    for model in imagenette_models:
        run_experiment(
            dataset_name="Imagenette",
            current_model_class=model,
            save_dir=f"{file_path}/{model.__name__}"
        )

    EXPERIMENT_MODE = 2
    file_path = f"eksperyment_{EXPERIMENT_MODE}"
    mnist_models = [MNIST_P1, MNIST_P1_2F]
    for model in mnist_models:
        run_experiment(
            dataset_name="MNIST",
            current_model_class=model,
            save_dir=f"{file_path}/{model.__name__}"
        )

    imagenette_models = [IMAGENETTE_P1, IMAGENETTE_P1_2F]
    for model in imagenette_models:
        run_experiment(
            dataset_name="Imagenette",
            current_model_class=model,
            save_dir=f"{file_path}/{model.__name__}"
        )
