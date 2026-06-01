import torch
from torchvision import datasets, transforms

# 1. Pobieramy zbiór treningowy.
# Używamy TYLKO ToTensor, aby zamienić piksele 0-255 na ułamki 0.0-1.0
temp_dataset = datasets.MNIST(root='./data', train=True, download=True,
                              transform=transforms.ToTensor())

# 2. Sklejamy wszystkie 60 000 obrazków do jednego potężnego tensora
# Wymiary to teraz (60000, 1, 28, 28)
loader = torch.utils.data.DataLoader(temp_dataset, batch_size=len(temp_dataset), shuffle=False)
data = next(iter(loader))

# data[0] zawiera wszystkie obrazy.
# Wywołujemy po prostu wbudowane matematyczne funkcje PyTorcha:
mean = data[0].mean()
std = data[0].std()

print(f"Wyliczona średnia (Mean): {mean.item():.4f}")
print(f"Wyliczone odchylenie (Std): {std.item():.4f}")