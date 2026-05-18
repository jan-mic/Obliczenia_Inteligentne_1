import os
import itertools
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Patch

from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    silhouette_score
)

from scipy.spatial import Voronoi

from datasets.loaders import load_dataset


def plot_voronoi(
    x,
    y_true,
    y_pred,
    digit_a,
    digit_b,
    dataset_name,
    save_path=None
):

    if x.shape[1] != 2:
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

    fig, ax = plt.subplots(figsize=(9, 7))

    class_colors = {
        digit_a: "royalblue",
        digit_b: "crimson"
    }

    region_colors = {
        0: "royalblue",
        1: "crimson"
    }

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
            color=region_colors[y_pred[index]],
            alpha=0.15,
            edgecolor="black",
            linewidth=0.3
        )

    point_colors = [
        class_colors[label]
        for label in y_true
    ]

    ax.scatter(
        x[:, 0],
        x[:, 1],
        c=point_colors,
        edgecolors="black",
        linewidths=0.3,
        s=20,
        alpha=0.9
    )

    legend_handles = [
        Patch(
            color="royalblue",
            label=f"Digit {digit_a}"
        ),
        Patch(
            color="crimson",
            label=f"Digit {digit_b}"
        )
    ]

    ax.legend(
        handles=legend_handles,
        title="Classes",
        loc="upper right"
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
        f"{dataset_name}: {digit_a} vs {digit_b}"
    )

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    plt.grid(True)

    if save_path:

        os.makedirs(
            os.path.dirname(save_path),
            exist_ok=True
        )

        plt.savefig(
            save_path,
            bbox_inches="tight",
            dpi=150
        )

    plt.close()


def analyze_pair_task(args):

    (
        x_train,
        y_train,
        digit_a,
        digit_b,
        dataset_name
    ) = args

    mask = np.logical_or(
        y_train == digit_a,
        y_train == digit_b
    )

    x_pair = x_train[mask]
    y_pair = y_train[mask]

    kmeans = KMeans(
        n_clusters=2,
        random_state=264321,
        n_init=10
    )

    clusters = kmeans.fit_predict(x_pair)

    ari = adjusted_rand_score(
        y_pair,
        clusters
    )

    silhouette = silhouette_score(
        x_pair,
        clusters,
        sample_size=2000,
        random_state=264321
    )

    if x_pair.shape[1] == 2:

        plot_voronoi(
            x_pair,
            y_pair,
            clusters,
            digit_a,
            digit_b,
            dataset_name,
            save_path=(
                f"plots/"
                f"{dataset_name}/"
                f"{digit_a}_vs_{digit_b}.png"
            )
        )

    log_text = (
        "\n--------------------------------\n"
        f"{digit_a} vs {digit_b}\n"
        "--------------------------------\n"
        f"ARI: {ari:.4f}\n"
        f"Silhouette: {silhouette:.4f}\n"
    )

    return {
        "digits": (
            digit_a,
            digit_b
        ),
        "ari": ari,
        "silhouette": silhouette,
        "log": log_text
    }


class PairwiseSeparabilityAnalyzer:

    def __init__(self, dataset_name):

        self.dataset_name = dataset_name

        train_dataset, test_dataset = load_dataset(
            dataset_name
        )

        self.x_train = train_dataset.x.numpy()
        self.y_train = train_dataset.y.numpy()

        self.x_test = test_dataset.x.numpy()
        self.y_test = test_dataset.y.numpy()

    def run(self):

        print("\n====================================")
        print(
            f"PAIRWISE ANALYSIS: "
            f"{self.dataset_name}"
        )
        print("====================================")

        unique_digits = np.unique(
            self.y_train
        )

        tasks = []

        for digit_a, digit_b in itertools.combinations(
            unique_digits,
            2
        ):

            tasks.append((
                self.x_train,
                self.y_train,
                digit_a,
                digit_b,
                self.dataset_name
            ))

        results = []

        with ProcessPoolExecutor(max_workers=6) as executor:

            futures = [
                executor.submit(
                    analyze_pair_task,
                    task
                )
                for task in tasks
            ]

            for future in as_completed(futures):

                result = future.result()

                print(result["log"])

                results.append(result)

        print("\n====================================")
        print("BEST SEPARATED PAIRS")
        print("====================================")

        best_results = sorted(
            results,
            key=lambda x: x["ari"],
            reverse=True
        )

        for result in best_results[:5]:

            print(
                f"{result['digits'][0]} "
                f"vs "
                f"{result['digits'][1]}"
                f" | "
                f"Silhouette: "
                f"{result['silhouette']:.4f}"
                f" | "
                f"ARI: "
                f"{result['ari']:.4f}"
            )

        print("\n====================================")
        print("WORST SEPARATED PAIRS")
        print("====================================")

        for result in best_results[-5:]:

            print(
                f"{result['digits'][0]} "
                f"vs "
                f"{result['digits'][1]}"
                f" | "
                f"Silhouette: "
                f"{result['silhouette']:.4f}"
                f" | "
                f"ARI: "
                f"{result['ari']:.4f}"
            )


def main():

    start_time = time.perf_counter()

    datasets = [
        "mnist_flatten",
        "mnist_density_center",
        "mnist_symmetry",
        "mnist_grid_feature",
        "mnist_projection_feature"
    ]

    for dataset_name in datasets:

        analyzer = PairwiseSeparabilityAnalyzer(
            dataset_name
        )

        analyzer.run()

    end_time = time.perf_counter()

    elapsed = end_time - start_time

    print("\n====================================")
    print(
        f"Total execution time: "
        f"{elapsed:.2f} seconds"
    )
    print("====================================")


if __name__ == "__main__":
    main()