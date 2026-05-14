import os
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    homogeneity_score,
    completeness_score,
    silhouette_score
)

from scipy.spatial import Voronoi

from datasets.loaders import load_dataset


def plot_voronoi(
    x,
    y_true,
    y_pred,
    title,
    save_path=None
):

    if x.shape[1] != 2:
        print(f"Skipping Voronoi for {title} (not 2D)")
        return

    min_x, min_y = x.min(axis=0)
    max_x, max_y = x.max(axis=0)

    max_point_value = max(
        abs(min_x),
        abs(min_y),
        abs(max_x),
        abs(max_y)
    )

    bounding_frame = np.array([
        [min_x - max_point_value, min_y - max_point_value],
        [min_x - max_point_value, max_y + max_point_value],
        [max_x + max_point_value, min_y - max_point_value],
        [max_x + max_point_value, max_y + max_point_value]
    ])

    x_with_bounding = np.vstack([
        x,
        bounding_frame
    ])

    vor = Voronoi(x_with_bounding)

    y_pred = y_pred.astype(int)

    fig, ax = plt.subplots(figsize=(9, 7))

    colors = [
        "orange",
        "green",
        "blue",
        "red",
        "yellow",
        "purple",
        "cyan",
        "magenta",
        "brown",
        "pink"
    ]

    colors_plasma = plt.colormaps["plasma"](
        np.linspace(
            0,
            1,
            len(np.unique(y_pred))
        )
    )

    colors_combined = (
        colors +
        colors_plasma.tolist()
    )

    cmap_combined = ListedColormap(
        colors_combined[:-10]
    )

    for index, point_region_id in enumerate(vor.point_region):

        if index >= len(x):
            break

        region = vor.regions[point_region_id]

        if -1 in region or len(region) == 0:
            continue

        polygon = [
            vor.vertices[i]
            for i in region
        ]

        ax.fill(
            *zip(*polygon),
            color=cmap_combined(
                y_pred[index]
            ),
            alpha=0.35,
            edgecolor="black",
            linewidth=0.5
        )

    n_classes = len(np.unique(y_true))
    true_cmap = plt.get_cmap("tab10", n_classes)

    ax.scatter(
        x[:, 0],
        x[:, 1],
        c=y_true,
        cmap=true_cmap,
        edgecolors="black",
        linewidths=0.3,
        s=18,
        alpha=0.9
    )

    padding = 0.5

    ax.set_xlim(
        min_x - padding,
        max_x + padding
    )

    ax.set_ylim(
        min_y - padding,
        max_y + padding
    )

    plt.title(
        f"Voronoi diagram: {title}",
        fontsize=14
    )

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    plt.grid(True)

    if save_path:
        plt.savefig(
            save_path,
            bbox_inches="tight",
            dpi=150
        )

    from matplotlib.patches import Patch

    n_classes = len(np.unique(y_true))
    true_cmap = plt.get_cmap("tab10", n_classes)

    legend_handles = [
        Patch(
            color=true_cmap(i),
            label=f"Digit {i}"
        )
        for i in range(n_classes)
    ]

    ax.legend(
        handles=legend_handles,
        title="MNIST classes",
        loc="upper right",
        fontsize=9
    )

    plt.show()


class SeparabilityAnalyzer:

    def __init__(self, dataset_name):

        self.dataset_name = dataset_name

        train_dataset, test_dataset = load_dataset(
            dataset_name
        )

        self.X_train = train_dataset.x.numpy()
        self.y_train = train_dataset.y.numpy()

        self.X_test = test_dataset.x.numpy()
        self.y_test = test_dataset.y.numpy()

    def analyze_kmeans(
        self,
        n_clusters=None
    ):

        if n_clusters is None:
            n_clusters = len(
                np.unique(self.y_train)
            )

        print("\n==============================")
        print(
            f"KMeans analysis: {self.dataset_name}"
        )
        print("==============================")

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=264321,
            n_init=10
        )

        clusters = kmeans.fit_predict(
            self.X_train
        )

        ari = adjusted_rand_score(
            self.y_train,
            clusters
        )

        homogeneity = homogeneity_score(
            self.y_train,
            clusters
        )

        completeness = completeness_score(
            self.y_train,
            clusters
        )

        silhouette = silhouette_score(
            self.X_train,
            clusters
        )

        print(f"ARI: {ari:.4f}")
        print(
            f"Homogeneity: "
            f"{homogeneity:.4f}"
        )

        print(
            f"Completeness: "
            f"{completeness:.4f}"
        )

        print(
            f"Silhouette: "
            f"{silhouette:.4f}"
        )

        return {
            "clusters": clusters,
            "ari": ari,
            "homogeneity": homogeneity,
            "completeness": completeness,
            "silhouette": silhouette
        }

    def run(self):

        print("\n====================================")
        print(f"DATASET: {self.dataset_name}")
        print("====================================")

        result = self.analyze_kmeans()

        if self.X_train.shape[1] == 2:

            os.makedirs(
                "plots",
                exist_ok=True
            )

            plot_voronoi(
                self.X_train,
                self.y_train,
                result["clusters"],
                self.dataset_name,
                save_path=(
                    f"plots/"
                    f"voronoi_"
                    f"{self.dataset_name}.png"
                )
            )

        else:
            print(
                f"Skipping Voronoi for "
                f"{self.dataset_name} "
                f"(not 2D)"
            )


def main():

    datasets = [
        "mnist_flatten",
        "mnist_density_center",
        "mnist_symmetry",
        "mnist_grid_feature",
        "mnist_projection_feature"
    ]

    for dataset_name in datasets:

        analyzer = SeparabilityAnalyzer(
            dataset_name
        )

        analyzer.run()


if __name__ == "__main__":
    main()