import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader
import numpy as np


def get_balanced_subset(dataset, total_samples):
    """Zwraca zbalansowany podzbiór danych (równa liczba klas)."""
    if hasattr(dataset, 'targets'):
        targets = np.array(dataset.targets)
    else:
        targets = np.array([sample[1] for sample in dataset])

    classes = np.unique(targets)
    samples_per_class = total_samples // len(classes)

    indices = []
    for c in classes:
        class_indices = np.where(targets == c)[0]
        chosen_indices = np.random.choice(class_indices, samples_per_class, replace=False)
        indices.extend(chosen_indices)

    return Subset(dataset, indices)


def compute_mean_std(dataset, batch_size=256):
    """
    Automatycznie oblicza średnią i odchylenie standardowe dla podanego zbioru danych.
    Działa iteracyjnie (batchami), aby nie wyczerpać pamięci RAM przy dużych zbiorach.
    """
    print("   [Trwa wyliczanie statystyk (mean, std) dla zbioru. Może to chwilę potrwać...]")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for data, _ in loader:
        # data to tensor o wymiarach: [batch_size, channels, height, width]
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    # Zwracamy jako tuple (krotki), żeby pasowało do transforms.Normalize
    return tuple(mean.tolist()), tuple(std.tolist())


def get_mnist_datasets():
    """Przygotowuje zbiór MNIST wraz z augmentacjami i dynamicznymi statystykami."""

    # 1. Ładujemy "surowy" zbiór tylko po to, żeby wyliczyć statystyki
    raw_transform = transforms.ToTensor()
    raw_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=raw_transform)

    # Wywołujemy naszą nową funkcję i drukujemy wynik
    mean, std = compute_mean_std(raw_train)
    print(f"   => Obliczona Średnia MNIST: {mean}")
    print(f"   => Obliczone Odchylenie MNIST: {std}")

    # 2. Definiujemy transformacje z wyliczonymi wartościami
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    aug_person1 = transforms.Compose([
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    aug_person2 = transforms.Compose([
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # 3. Tworzymy ostateczne zbiory danych z nałożonymi transformacjami
    train_base = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=base_transform)
    train_aug1 = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=aug_person1)
    train_aug2 = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=aug_person2)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=base_transform)

    class_names = [str(i) for i in range(10)]

    return train_base, train_aug1, train_aug2, test_dataset, class_names, 1


def get_imagenette_datasets():
    """Przygotowuje zbiór Imagenette wraz z augmentacjami i dynamicznymi statystykami."""

    # 1. Ładujemy surowy zbiór - uwaga: przy Imagenette musimy ujednolicić rozmiar przed liczeniem statystyk!
    raw_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    raw_train = torchvision.datasets.Imagenette(root='./data', split='train', download=True, transform=raw_transform)

    # Wywołujemy nową funkcję i drukujemy wynik
    mean, std = compute_mean_std(raw_train)
    print(f"   => Obliczona Średnia Imagenette: {mean}")
    print(f"   => Obliczone Odchylenie Imagenette: {std}")

    # 2. Definiujemy transformacje z wyliczonymi wartościami
    base_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    aug_person1 = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    aug_person2 = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # 3. Tworzymy ostateczne zbiory danych
    train_base = torchvision.datasets.Imagenette(root='./data', split='train', download=True, transform=base_transform)
    train_aug1 = torchvision.datasets.Imagenette(root='./data', split='train', download=True, transform=aug_person1)
    train_aug2 = torchvision.datasets.Imagenette(root='./data', split='train', download=True, transform=aug_person2)
    test_dataset = torchvision.datasets.Imagenette(root='./data', split='val', download=True, transform=base_transform)

    class_names = ['tench', 'English springer', 'cassette player', 'chain saw', 'church',
                   'French horn', 'garbage truck', 'gas pump', 'golf ball', 'parachute']

    return train_base, train_aug1, train_aug2, test_dataset, class_names, 3