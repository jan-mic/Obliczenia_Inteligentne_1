import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
import numpy as np


def get_balanced_subset(dataset, total_samples):
    """Zwraca zbalansowany podzbiór danych (równa liczba klas)."""
    # Dla torchvision.datasets etykiety mogą być pod 'targets' lub trzeba je wyciągnąć iteracyjnie
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


def get_mnist_datasets():
    """Przygotowuje zbiór MNIST wraz z augmentacjami."""
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    aug_person1 = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    aug_person2 = transforms.Compose([
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_base = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=base_transform)
    train_aug1 = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=aug_person1)
    train_aug2 = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=aug_person2)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=base_transform)

    class_names = [str(i) for i in range(10)]

    return train_base, train_aug1, train_aug2, test_dataset, class_names, 1  # 1 to liczba kanałów (input_channels)


def get_imagenette_datasets():
    """Przygotowuje zbiór Imagenette wraz z augmentacjami."""
    base_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    aug_person1 = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    aug_person2 = transforms.Compose([
        transforms.Resize((136, 136)),
        transforms.RandomCrop((128, 128)),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_base = torchvision.datasets.Imagenette(root='./data', split='train', download=True, transform=base_transform)
    train_aug1 = torchvision.datasets.Imagenette(root='./data', split='train', download=True, transform=aug_person1)
    train_aug2 = torchvision.datasets.Imagenette(root='./data', split='train', download=True, transform=aug_person2)
    test_dataset = torchvision.datasets.Imagenette(root='./data', split='val', download=True, transform=base_transform)

    class_names = ['bench', 'English springer', 'cassette player', 'chain saw', 'church',
                   'French horn', 'garbage truck', 'gas pump', 'golf ball', 'parachute']

    return train_base, train_aug1, train_aug2, test_dataset, class_names, 3  # 3 to liczba kanałów (RGB)