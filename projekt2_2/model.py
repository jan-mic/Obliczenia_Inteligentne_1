import torch.nn as nn


class CNN_Standard_MNIST(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_Standard_MNIST, self).__init__()
        # Wejście: 1 kanał (Grayscale), rozmiar 28x28
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Rozmiar: 14x14
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # Rozmiar: 7x7
        )
        self.flatten = nn.Flatten()

        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


class CNN_2Features_MNIST(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_2Features_MNIST, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.flatten = nn.Flatten()

        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # WĄSKIE GARDŁO: 2 cechy
            nn.Linear(2, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


# --- ARCHITEKTURY DLA ZBIORU IMAGENETTE ---

class CNN_Standard_Imagenette(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_Standard_Imagenette, self).__init__()
        # Wejście: 3 kanały (RGB), rozmiar 128x128
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Rozmiar: 64x64
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Rozmiar: 32x32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # Rozmiar: 16x16
        )
        self.flatten = nn.Flatten()

        # Klasyfikator (MLP) [cite: 6]
        self.classifier = nn.Sequential(
            nn.Linear(64 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout dla zapobiegania przeuczeniu [cite: 18]
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


class CNN_2Features_Imagenette(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_2Features_Imagenette, self).__init__()
        # Architektura z wymuszonym wektorem 2 cech [cite: 41, 61]
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.flatten = nn.Flatten()

        # Klasyfikator z "wąskim gardłem"
        self.classifier = nn.Sequential(
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 2),  # WĄSKIE GARDŁO: 2 cechy [cite: 41, 61]
            nn.Linear(2, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x