import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

wine = load_wine()
X = wine.data

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k = 5

neighbors = NearestNeighbors(n_neighbors=k)
neighbors_fit = neighbors.fit(X_scaled)

distances, indices = neighbors_fit.kneighbors(X_scaled)

distances = np.sort(distances[:, k - 1], axis=0)

plt.figure(figsize=(10, 6))
plt.plot(distances, linewidth=2, color='darkblue')
plt.title('Wykres K-odległości (Metoda Łokcia) dla zbioru WINE')
plt.xlabel('Punkty danych (posortowane wg odległości)')
plt.ylabel(f'Odległość do {k}-tego sąsiada (Eps)')
plt.grid(True, linestyle='--', alpha=0.7)

# Opcjonalnie: ułatwienie podziałki
plt.yticks(np.arange(0, max(distances) + 0.5, 0.5))

plt.show()
