import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from matplotlib.patches import Patch
from sklearn.cluster import KMeans, DBSCAN

from sklearn.metrics import (
    adjusted_rand_score,
    homogeneity_score,
    completeness_score,
    silhouette_score
)

from scipy.spatial import Voronoi

from datasets.loaders import load_dataset
from models.mlp import MLP


MODEL_MAP = {
    "mnist_flatten": "mnist_flatten.pth",
    "mnist_density_center": "mnist_density_center.pth",
    "mnist_symmetry": "mnist_symmetry.pth",
    "mnist_grid_feature": "mnist_grid_feature.pth",
    "mnist_projection_feature": "mnist_projection_feature.pth",
    "iris": "iris.pth",
    "wine": "wine.pth"
}


def load_model(dataset_name):
    path = os.path.join("saved_models", MODEL_MAP[dataset_name])

    checkpoint = torch.load(path, map_location="cpu")

    model = MLP(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_classes=checkpoint["num_classes"]
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def plot_metrics_bar(results, save_path=None):
    datasets = list(results.keys())

    ari = [results[d]["ari"] for d in datasets]
    homogeneity = [results[d]["homogeneity"] for d in datasets]
    completeness = [results[d]["completeness"] for d in datasets]

    x = np.arange(len(datasets))
    width = 0.2

    plt.figure(figsize=(12, 6))

    plt.bar(x - 1.5 * width, ari, width, label="ARI")
    plt.bar(x - 0.5 * width, homogeneity, width, label="Homogeneity")
    plt.bar(x + 0.5 * width, completeness, width, label="Completeness")

    plt.xticks(x, datasets, rotation=30, ha="right")
    plt.ylabel("Score")
    plt.title("Clustering quality comparison (Model predictions)")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)

    plt.show()


def plot_voronoi(x, y_true, y_pred, title, save_path=None):

    if x.shape[1] != 2:
        return

    min_x, min_y = x.min(axis=0)
    max_x, max_y = x.max(axis=0)

    max_point_value = max(abs(min_x), abs(min_y), abs(max_x), abs(max_y))

    bounding_frame = np.array([
        [min_x - max_point_value, min_y - max_point_value],
        [min_x - max_point_value, max_y + max_point_value],
        [max_x + max_point_value, min_y - max_point_value],
        [max_x + max_point_value, max_y + max_point_value]
    ])

    x_with_bounding = np.vstack([x, bounding_frame])

    vor = Voronoi(x_with_bounding)

    y_pred = y_pred.astype(int)

    fig, ax = plt.subplots(figsize=(9, 7))

    n_classes = len(np.unique(y_true))
    cmap = plt.get_cmap("tab10", n_classes)

    for index, point_region_id in enumerate(vor.point_region):

        if index >= len(x):
            break

        region = vor.regions[point_region_id]

        if -1 in region or len(region) == 0:
            continue

        polygon = [vor.vertices[i] for i in region]

        ax.fill(
            *zip(*polygon),
            color=cmap(y_pred[index]),
            alpha=0.35,
            edgecolor="black",
            linewidth=0.5
        )

    ax.scatter(
        x[:, 0],
        x[:, 1],
        c=y_true,
        cmap=cmap,
        edgecolors="black",
        linewidths=0.3,
        s=18,
        alpha=0.9
    )

    ax.set_xlim(min_x - 0.5, max_x + 0.5)
    ax.set_ylim(min_y - 0.5, max_y + 0.5)

    plt.title(f"Voronoi diagram: {title}")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True)

    legend_handles = [
        Patch(color=cmap(i), label=f"Class {i}")
        for i in range(n_classes)
    ]

    ax.legend(handles=legend_handles, loc="upper right", fontsize=9)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)

    plt.show()


class SeparabilityAnalyzer:

    def __init__(self, dataset_name, model):
        self.dataset_name = dataset_name
        self.model = model

        train_dataset, _ = load_dataset(dataset_name)

        self.X_train = train_dataset.x.numpy()
        self.y_train = train_dataset.y.numpy()

    def predict(self):
        with torch.no_grad():
            X = torch.tensor(self.X_train, dtype=torch.float32)
            logits = self.model(X)
            preds = torch.argmax(logits, dim=1)
            return preds.numpy()

    def analyze_model(self):

        y_pred = self.predict()

        ari = adjusted_rand_score(self.y_train, y_pred)
        homogeneity = homogeneity_score(self.y_train, y_pred)
        completeness = completeness_score(self.y_train, y_pred)

        return {
            "clusters": y_pred,
            "ari": ari,
            "homogeneity": homogeneity,
            "completeness": completeness,
        }

    def analyze_model2(self):

        n_clusters = len(np.unique(self.y_train))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        y_pred_kmeans = kmeans.fit_predict(self.X_train)

        kmeans_ari = adjusted_rand_score(self.y_train, y_pred_kmeans)
        kmeans_homo = homogeneity_score(self.y_train, y_pred_kmeans)
        kmeans_comp = completeness_score(self.y_train, y_pred_kmeans)
        kmeans_sil = silhouette_score(self.X_train, y_pred_kmeans, random_state=42)

        print(f"K-Means -> ARI: {kmeans_ari:.4f} | Homo: {kmeans_homo:.4f} | Comp: {kmeans_comp:.4f}")

        ##
        dbscan = DBSCAN(eps=0.5, min_samples=1)
        y_pred_dbscan = dbscan.fit_predict(self.X_train)

        dbscan_ari = adjusted_rand_score(self.y_train, y_pred_dbscan)
        dbscan_homo = homogeneity_score(self.y_train, y_pred_dbscan)
        dbscan_comp = completeness_score(self.y_train, y_pred_dbscan)
        dbscan_sil = silhouette_score(self.X_train, y_pred_dbscan, random_state=42)

        dbscan_found_clusters = len(set(y_pred_dbscan)) - (1 if -1 in y_pred_dbscan else 0)
        print(f"DBSCAN  -> Znaleziono klastrów: {dbscan_found_clusters}")
        print(f"DBSCAN  -> ARI: {dbscan_ari:.4f} | Homo: {dbscan_homo:.4f} | Comp: {dbscan_comp:.4f}")

        return {
            "clusters": y_pred_kmeans,
            "ari": kmeans_ari,
            "homogeneity": kmeans_homo,
            "completeness": kmeans_comp,
        }

    def run(self):

        result = self.analyze_model()

        if self.X_train.shape[1] == 2:
            os.makedirs("plots", exist_ok=True)

            plot_voronoi(
                self.X_train,
                self.y_train,
                result["clusters"],
                self.dataset_name,
                save_path=f"plots/voronoi_{self.dataset_name}.png"
            )


def main():

    datasets = [
        "mnist_flatten",
        # "mnist_density_center",
        "mnist_symmetry",
        "mnist_projection_feature",
        # "mnist_grid_feature",
        # "iris",
        # "wine"
    ]

    results = {}

    for dataset_name in datasets:

        if dataset_name not in MODEL_MAP:
            continue

        model = load_model(dataset_name)

        analyzer = SeparabilityAnalyzer(dataset_name, model)
        result = analyzer.analyze_model()

        results[dataset_name] = result

        if analyzer.X_train.shape[1] == 2:
            os.makedirs("plots", exist_ok=True)

            plot_voronoi(
                analyzer.X_train,
                analyzer.y_train,
                result["clusters"],
                dataset_name,
                save_path=f"plots/voronoi_{dataset_name}.png"
            )

    plot_metrics_bar(
        results,
        save_path="plots/metric_bar_model_comparison_grid_density.png"
    )


if __name__ == "__main__":
    main()