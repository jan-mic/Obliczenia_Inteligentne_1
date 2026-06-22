import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
import os


def plot_augmentations(dataset, index, aug1, aug2, is_mnist=False):
    """
    Pobiera obraz pod wskazanym indeksem ze zbioru, aplikuje dwie
    różne augmentacje i wyświetla je obok siebie.
    """
    # 1. Pobranie czystego obrazka (PIL Image) i etykiety
    img, label = dataset[index]

    # 2. Aplikacja augmentacji
    img_aug1 = aug1(img)
    img_aug2 = aug2(img)

    # 3. Przygotowanie płótna do rysowania
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles = ['oryginał', 'aug1', 'aug2']
    images = [img, img_aug1, img_aug2]

    for ax, image, title in zip(axes, images, titles):
        # Konwersja PIL Image na Tensor w celu ujednolicenia wyświetlania
        if not isinstance(image, torch.Tensor):
            image = TF.to_tensor(image)

        # Zamiana z formatu PyTorch (C, H, W) na format Matplotlib (H, W, C)
        img_np = image.permute(1, 2, 0).numpy()

        # Rysowanie z odpowiednią mapą kolorów
        if is_mnist:
            ax.imshow(img_np.squeeze(), cmap='gray')
        else:
            ax.imshow(img_np)

        ax.set_title(title, fontsize=14)
        ax.axis('off')  # Ukrywamy osie i wartości liczbowe

    plt.suptitle(f"Obraz indeks: {index} | Klasa: {label}", fontsize=16)
    plt.tight_layout()
    plt.show()


# ==========================================
# SEKCJA 1: ZBIÓR MNIST
# ==========================================
print("Ładowanie zbioru MNIST...")
# Ładujemy zbiór BEZ ToTensor(), aby mieć czyste obrazy PIL do manipulacji
mnist_data = datasets.MNIST(root='./data', train=True, download=True)

# Definicje augmentacji z Twojego raportu (Strona 5)
mnist_aug1 = transforms.RandomRotation(degrees=[-20, 20])
mnist_aug2 = transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))

# Wybierz indeksy dla MNIST, które chcesz podejrzeć (np. 0, 1, 2)
indeksy_mnist = [0, 15, 42]
for idx in indeksy_mnist:
    plot_augmentations(mnist_data, idx, mnist_aug1, mnist_aug2, is_mnist=True)

# ==========================================
# SEKCJA 2: ZBIÓR IMAGENETTE
# ==========================================
print("Ładowanie zbioru Imagenette...")
# Pobieranie Imagenette. W zależności od wersji torchvision używamy
# wbudowanej klasy lub ładujemy z folderu. Używamy wbudowanej klasy:
try:
    # Resize(128) aby odwzorować rozmiar z Twojego projektu przed augmentacjami
    base_transform = transforms.Resize((128, 128))
    imagenette_data = datasets.Imagenette(root='./data', split='train',
                                          download=True, transform=base_transform)

    # Definicje augmentacji z Twojego raportu (Strona 6)
    # p=1.0 wymusza odbicie w 100% przypadków dla celów samej wizualizacji
    imagenette_aug1 = transforms.RandomHorizontalFlip(p=1.0)
    imagenette_aug2 = transforms.RandomRotation(degrees=[-20, 20])

    # Wybierz indeksy dla Imagenette
    indeksy_imagenette = [0, 150, 3000]
    for idx in indeksy_imagenette:
        plot_augmentations(imagenette_data, idx, imagenette_aug1, imagenette_aug2, is_mnist=False)

except AttributeError:
    print("Twoja wersja PyTorch nie ma wbudowanej klasy Imagenette.")
    print("Jeśli masz go na dysku w folderze, użyj datasets.ImageFolder('sciezka/do/imagenette')")