import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.metrics import silhouette_score, adjusted_rand_score, homogeneity_score, completeness_score
from scipy.spatial import Voronoi
from matplotlib import colormaps
from sklearn.preprocessing import StandardScaler


def load(path):
    data = np.loadtxt(path, delimiter=';')
    x = data[:, :-1]
    y = data[:, -1]
    return x, y


def method_k_means(X, y_true, file_i, file_j):
    X = StandardScaler().fit_transform(X)
    range_n_clusters = range(2, 10)
    silhouette_scores = []
    ars_values = []
    hs_values = []
    cs_values = []
    average_scores = []

    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=n_clusters)
        y_pred = kmeans.fit_predict(X)

        score = silhouette_score(X, y_pred)
        silhouette_scores.append(score)
        ars = adjusted_rand_score(y_true, y_pred)
        hs = homogeneity_score(y_true, y_pred)
        cs = completeness_score(y_true, y_pred)
        average_score = (ars + hs + cs) / 3

        ars_values.append(ars)
        hs_values.append(hs)
        cs_values.append(cs)
        average_scores.append(average_score)

    silhouette_scores = np.array(silhouette_scores)
    average_scores = np.array(average_scores)

    draw_experiment_1_kmeans(range_n_clusters, silhouette_scores, file_i, file_j)
    draw_experiment_2(range_n_clusters, ars_values, hs_values, cs_values, file_i, file_j, 'K-Means')

    if file_i != 0:
        info_best = f'Silhouette score: {silhouette_scores.max().round(3)} (best) | Clusters: {range_n_clusters[silhouette_scores.argmax()]}'
        info_worst = f'Silhouette score: {silhouette_scores.min().round(3)} (worst) | Clusters: {range_n_clusters[silhouette_scores.argmin()]}'
        info_average_best = f'Avg. score: {average_scores.max().round(3)} (best) | Clusters: {range_n_clusters[average_scores.argmax()]}'
        info_average_worst = f'Avg. score: {average_scores.min().round(3)} (worst) | Clusters: {range_n_clusters[average_scores.argmin()]}'

        draw_voronoi_diagram_kmeans(range_n_clusters[silhouette_scores.argmax()], X, y_true, file_i, file_j, info_best,
                                    "best_sil")
        draw_voronoi_diagram_kmeans(range_n_clusters[silhouette_scores.argmin()], X, y_true, file_i, file_j, info_worst,
                                    "worst_sil")

        draw_voronoi_diagram_kmeans(range_n_clusters[average_scores.argmax()], X, y_true, file_i, file_j,
                                    info_average_best, "best_avg")
        draw_voronoi_diagram_kmeans(range_n_clusters[average_scores.argmin()], X, y_true, file_i, file_j,
                                    info_average_worst, "worst_avg")
    else:
        print(f'>>> Data: {file_j} | Method: K-Means <<<')
        print('Experiment: 1')
        print(
            f'Best Silhouette score: {silhouette_scores.max().round(3)} | Clusters: {range_n_clusters[silhouette_scores.argmax()]}')
        print(
            f'Worst Silhouette score: {silhouette_scores.min().round(3)} | Clusters: {range_n_clusters[silhouette_scores.argmin()]}')
        print('Experiment: 2')
        print(
            f'Best Average Score: {average_scores.max().round(3)} | Clusters: {range_n_clusters[average_scores.argmax()]}')
        print(
            f'Worst Average Score: {average_scores.min().round(3)} | Clusters: {range_n_clusters[average_scores.argmin()]}')
        print('=====================================================')


def draw_voronoi_diagram_kmeans(n_clusters, X, y_true, file_i, file_j, info, suffix):
    kmeans = KMeans(n_clusters=n_clusters)
    y_pred = kmeans.fit_predict(X)
    plot_voronoi_diagram(X, y_true, y_pred, file_i, file_j, info, f"kmeans_{suffix}")


def draw_experiment_1_kmeans(range_n_clusters, silhouette_scores, file_i, file_j):
    plt.plot(range_n_clusters, silhouette_scores, marker='o', markersize=3)
    if file_i == 0:
        plt.title(f'Experiment: 1 | Method: K-means | Data: {file_j}')
    else:
        plt.title(f'Experiment: 1 | Method: K-means | Data: {file_i}_{file_j}.csv')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette score')
    plt.ylim(0, 1.05)
    plt.legend(['silhouette'])
    plt.grid(True)
    title_suffix = f'{file_j}' if file_i == 0 else f'{file_i}_{file_j}'
    plt.savefig(f'wykresy/exp1_kmeans_{title_suffix}.png', bbox_inches='tight')
    plt.close()


def method_dbscan(X, y_true, file_i, file_j):
    X = StandardScaler().fit_transform(X)
    silhouette_scores = []
    n_clusters_list = []

    eps_values = np.arange(0.01, 1.7, 0.05)
    if file_j == 'Wine':
        eps_values = np.arange(1.0, 3.0, 0.05)
    if file_j == 'Breast cancer':
        eps_values = np.arange(1.2, 3.7, 0.05)

    ars_values = []
    hs_values = []
    cs_values = []
    average_scores = []

    for eps in eps_values:
        dbscan = DBSCAN(eps=eps, min_samples=1)  # min_samples przykładowo 5
        y_pred = dbscan.fit_predict(X)

        unique_labels = set(y_pred)
        if -1 in unique_labels:
            unique_labels.remove(-1)

        n_clusters = len(unique_labels)
        if len(X) > n_clusters >= 2:
            score = silhouette_score(X, y_pred)
        else:
            score = 0

        ars = adjusted_rand_score(y_true, y_pred)
        hs = homogeneity_score(y_true, y_pred)
        cs = completeness_score(y_true, y_pred)
        average_score = (ars + hs + cs) / 3

        silhouette_scores.append(score)
        n_clusters_list.append(n_clusters)
        ars_values.append(ars)
        hs_values.append(hs)
        cs_values.append(cs)
        average_scores.append(average_score)

    silhouette_scores = np.array(silhouette_scores)
    average_scores = np.array(average_scores)

    draw_experiment_1_dbscan(eps_values, silhouette_scores, n_clusters_list, file_i, file_j)
    draw_experiment_2(eps_values, ars_values, hs_values, cs_values, file_i, file_j, 'DBSCAN')

    if file_i != 0:
        info_best = f"Eps: {eps_values[silhouette_scores.argmax()].round(3)} (best) | silhouette: {silhouette_scores.max().round(3)} | Clusters: {n_clusters_list[silhouette_scores.argmax()]}"
        info_worst = f"Eps: {eps_values[silhouette_scores.argmin()].round(3)} (worst) | silhouette: {silhouette_scores.min().round(3)} | Clusters: {n_clusters_list[silhouette_scores.argmin()]}"
        info_average_best = f'Avg. score: {average_scores.max().round(3)} (best) | Eps: {eps_values[average_scores.argmax()].round(3)} | Clusters: {n_clusters_list[average_scores.argmax()]}'
        info_average_worst = f'Avg. score: {average_scores.min().round(3)} (worst) | Eps: {eps_values[average_scores.argmin()].round(3)} | Clusters: {n_clusters_list[average_scores.argmin()]}'

        draw_voronoi_diagram_dbscan(eps_values[silhouette_scores.argmax()], X, y_true, file_i, file_j, info_best,
                                    "best_sil")
        draw_voronoi_diagram_dbscan(eps_values[silhouette_scores.argmin()], X, y_true, file_i, file_j, info_worst,
                                    "worst_sil")

        draw_voronoi_diagram_dbscan(eps_values[average_scores.argmax()], X, y_true, file_i, file_j, info_average_best,
                                    "best_avg")
        draw_voronoi_diagram_dbscan(eps_values[average_scores.argmin()], X, y_true, file_i, file_j, info_average_worst,
                                    "worst_avg")
    else:
        print(f'>>> Data: {file_j} | Method: DBSCAN <<<')
        print('Experiment: 1')
        print(
            f'Best Silhouette score: {silhouette_scores.max().round(3)} Epsilon: {eps_values[silhouette_scores.argmax()].round(3)} | Clusters: {n_clusters_list[silhouette_scores.argmax()]}')
        print(
            f'Worst Silhouette score: {silhouette_scores.min().round(3)} Epsilon: {eps_values[silhouette_scores.argmin()].round(3)} | Clusters: {n_clusters_list[silhouette_scores.argmin()]}')

        print('Experiment: 2')
        print(
            f'Best Average Score: {average_scores.max().round(3)} Epsilon: {eps_values[average_scores.argmax()].round(3)} | Clusters: {n_clusters_list[average_scores.argmax()]}')
        print(
            f'Worst Average Score: {average_scores.min().round(3)} Epsilon: {eps_values[average_scores.argmin()].round(3)} | Clusters: {n_clusters_list[average_scores.argmin()]}')
        print('=====================================================')


def draw_voronoi_diagram_dbscan(eps, X, y_true, file_i, file_j, info, suffix):
    dbscan = DBSCAN(eps=eps, min_samples=1)
    y_pred = dbscan.fit_predict(X)
    plot_voronoi_diagram(X, y_true, y_pred, file_i, file_j, info, f"dbscan_{suffix}")


def draw_experiment_1_dbscan(eps_values, silhouette_scores, n_clusters_list, file_i, file_j):
    plt.plot(eps_values, silhouette_scores, marker='o', markersize=3)

    for eps in eps_values:
        plt.axvline(x=eps, color='gray', linestyle='--', alpha=0.5)

    for i, (eps, score) in enumerate(zip(eps_values, silhouette_scores)):
        if i < len(n_clusters_list):
            plt.text(eps, score + 0.01,
                     str(n_clusters_list[i]),
                     fontsize=7)

    if file_i == 0:
        plt.title(f'Experiment: 1 | Method: DBSCAN | Data: {file_j}', fontsize=10)
    else:
        plt.title(f'Experiment: 1 | Method: DBSCAN | Data: {file_i}_{file_j}', fontsize=10)
    plt.xlabel('Eps')
    plt.ylabel('Silhouette score')
    plt.legend(['silhouette'])
    title_suffix = f'{file_j}' if file_i == 0 else f'{file_i}_{file_j}'
    plt.savefig(f'wykresy/exp1_dbscan{title_suffix}.png', bbox_inches='tight')
    plt.close()


def draw_experiment_2(x_values, ars_values, hs_values, cs_values, file_i, file_j, method):
    plt.figure(figsize=(8, 6))
    plt.plot(x_values, ars_values, marker='o', markersize=3, label='Adjusted Rand')
    plt.plot(x_values, hs_values, marker='o', markersize=3, label='Homogeneity')
    plt.plot(x_values, cs_values, marker='o', markersize=3, label='Completeness')

    if file_i == 0:
        plt.title(f'Experiment: 2 | Method: {method} | Data: {file_j}')
    else:
        plt.title(f'Experiment: 2 | Method: {method} | Data: {file_i}_{file_j}.csv')
    if method == 'K-Means':
        plt.xlabel('Number of clusters')
    if method == 'DBSCAN':
        plt.xlabel('Eps')
    plt.ylabel('Wartość')
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.grid(True)
    title_suffix = f'{file_j}' if file_i == 0 else f'{file_i}_{file_j}'
    plt.savefig(f'wykresy/exp2_{method}_{title_suffix}.png', bbox_inches='tight')
    plt.close()

def plot_voronoi_diagram(X, y_true, y_pred, file_i, file_j, info, suffix):
    min_x, min_y = X.min(axis=0)
    max_x, max_y = X.max(axis=0)
    max_point_value = max(abs(min_x), abs(min_y), abs(max_x), abs(max_y))
    bounding_frame = np.array([
        [min_x - max_point_value, min_y - max_point_value],
        [min_x - max_point_value, max_y + max_point_value],
        [max_x + max_point_value, min_y - max_point_value],
        [max_x + max_point_value, max_y + max_point_value]
    ])

    X_with_bounding = np.vstack([X, bounding_frame])
    vor = Voronoi(X_with_bounding)
    y_pred = y_pred.astype(int)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['orange', 'green', 'blue', 'red', 'yellow', 'purple', 'cyan', 'magenta', 'brown', 'pink']
    colors_plasma = colormaps['plasma'](np.linspace(0, 1, len(np.unique(y_pred))))
    colors_combined = colors + colors_plasma.tolist()
    cmap_combined = ListedColormap(colors_combined[:-10])

    for index, point_region_id in enumerate(vor.point_region):
        region = vor.regions[point_region_id]
        if not -1 in region:
            polygon = [vor.vertices[i] for i in region]
            ax.fill(*zip(*polygon), color=cmap_combined(y_pred[index]), alpha=0.4, edgecolor='k')

    ax.scatter(X[:, 0], X[:, 1], c=y_true, cmap='rainbow', edgecolors='k', s=25)
    padding = 0.5
    ax.set_xlim(min_x - padding, max_x + padding)
    ax.set_ylim(min_y - padding, max_y + padding)
    plt.title(f"Data: {file_i}_{file_j}.csv | {info}", fontsize=14)
    title_suffix = f'{file_j}' if file_i == 0 else f'{file_i}_{file_j}'
    plt.savefig(f'wykresy/voronoi_{suffix}_{title_suffix}.png', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    os.makedirs('wykresy', exist_ok=True)

    for i in range(2, 4):
        for j in range(1, 4):
            X, y_true = load(f'../data/{i}_{j}.csv')
            method_k_means(X, y_true, i, j)
            method_dbscan(X, y_true, i, j)

    iris = load_iris()
    wine = load_wine()

    method_k_means(iris.data, iris.target, 0, 'Iris')
    method_dbscan(iris.data, iris.target, 0, 'Iris')

    method_k_means(wine.data, wine.target, 0, 'Wine')
    method_dbscan(wine.data, wine.target, 0, 'Wine')

    print(f"Unikalne etykiety: {iris.target_names}")
    print(f"Unikalne etykiety: {wine.target_names}")
