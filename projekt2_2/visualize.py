import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
import numpy as np
from torchvision.utils import save_image


def plot_accuracy_history(train_history, test_history, dataset_name, save_dir="scores"):
    epochs = range(1, len(train_history) + 1)
    plt.figure(figsize=(10, 6))

    plt.plot(epochs, train_history, label='Zbior treningowy', marker='o')
    plt.plot(epochs, test_history, label='Zbior testowy', marker='s')

    plt.title(f'Accuracy ({dataset_name})')
    plt.xlabel('Epoka')
    plt.xticks(epochs)
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    target_dir = os.path.join(save_dir, 'charts_exp_1')
    os.makedirs(target_dir, exist_ok=True)
    plt.savefig(os.path.join(target_dir, f"accuracy_{dataset_name}.png"))
    plt.show()


def plot_confusion_matrix(model, data_loader, device, class_names=None, dataset_name_cm='', save_dir="scores"):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)
    plt.title('Confusion Matrix', pad=20)
    plt.tight_layout()

    target_dir = os.path.join(save_dir, 'charts_exp_1')
    os.makedirs(target_dir, exist_ok=True)
    plt.savefig(os.path.join(target_dir, f"{dataset_name_cm}.png"))
    plt.show()


def plot_decision_boundaries(model, data_loader, device, dataset_name="", save_dir="scores"):
    model.eval()
    features_2d = []
    labels_list = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            x = model.features(inputs)
            x = model.flatten(x)

            for layer in list(model.classifier.children())[:-1]:
                x = layer(x)

            features_2d.append(x.cpu().numpy())
            labels_list.append(labels.numpy())

    features_2d = np.vstack(features_2d)
    labels_list = np.concatenate(labels_list)

    x_min, x_max = features_2d[:, 0].min() - 1, features_2d[:, 0].max() + 1
    y_min, y_max = features_2d[:, 1].min() - 1, features_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    grid_tensor = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]).to(device)
    last_layer = list(model.classifier.children())[-1]

    with torch.no_grad():
        Z = last_layer(grid_tensor)
        _, preds = torch.max(Z, 1)
        Z = preds.cpu().numpy().reshape(xx.shape)

    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.tab10)
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels_list,
                          cmap=plt.cm.tab10, edgecolor='k', s=20)
    plt.legend(*scatter.legend_elements(), title="Klasy", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('Granica decyzyjna')
    plt.xlabel("Cecha 1")
    plt.ylabel("Cecha 2")
    plt.tight_layout()

    target_dir = os.path.join(save_dir, 'charts_exp_1')
    os.makedirs(target_dir, exist_ok=True)
    plt.savefig(os.path.join(target_dir, f"{dataset_name}.png"))
    plt.show()


def save_augmentation_samples(original_dataset, aug_dataset, aug_name, dataset_name, num_samples=3, save_dir="scores"):
    target_dir = os.path.join(save_dir, 'images')
    os.makedirs(target_dir, exist_ok=True)

    for i in range(num_samples):
        orig_img, _ = original_dataset[i]
        aug_img, _ = aug_dataset[i]

        if orig_img.shape[0] == 3:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        else:
            mean = torch.tensor([0.1307]).view(1, 1, 1)
            std = torch.tensor([0.3081]).view(1, 1, 1)

        orig_img = orig_img * std + mean
        aug_img = aug_img * std + mean
        orig_img = orig_img.clamp(0, 1)
        aug_img = aug_img.clamp(0, 1)

        orig_filename = os.path.join(target_dir, f"{dataset_name}_probka{i + 1}_1_Oryginal.png")
        aug_filename = os.path.join(target_dir, f"{dataset_name}_probka{i + 1}_2_{aug_name}.png")
        save_image(orig_img, orig_filename)
        save_image(aug_img, aug_filename)
