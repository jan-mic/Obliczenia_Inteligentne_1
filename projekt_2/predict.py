import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from datasets.loaders import load_dataset
from models.mlp import MLP

OUTPUT_DIR = "results"

class Predictor:

    def __init__(self, dataset_name, model_path):

        self.dataset_name = dataset_name
        self.model_path = model_path

        train, test = load_dataset(dataset_name)

        self.X_test = torch.tensor(test.x.numpy(), dtype=torch.float32)
        self.y_test = torch.tensor(test.y.numpy(), dtype=torch.long)

        os.makedirs(OUTPUT_DIR, exist_ok=True)

    def load_model(self):
        path = os.path.join("saved_models", self.model_path)

        checkpoint = torch.load(path, map_location="cpu")

        model = MLP(
            input_dim=checkpoint["input_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            num_classes=checkpoint["num_classes"]
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        return model

    def predict(self, model):

        with torch.no_grad():
            logits = model(self.X_test)
            y_pred = torch.argmax(logits, dim=1)

        return y_pred.numpy()

    def evaluate(self, y_pred):

        y_true = self.y_test.numpy()

        acc = accuracy_score(y_true, y_pred)

        print("\n====================================")
        print(f"DATASET: {self.dataset_name}")
        print("====================================")

        print(f"Accuracy: {acc:.4f}")

        print("\nClassification report:")
        print(classification_report(y_true, y_pred))

        return y_true, y_pred

    def save_confusion_matrix(self, y_true, y_pred):

        cm = confusion_matrix(y_true, y_pred)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm)

        fig, ax = plt.subplots(figsize=(8, 8))

        disp.plot(
            ax=ax,
            cmap="Blues",
            values_format="d",
            colorbar=True
        )

        plt.title(f"Confusion Matrix: {self.dataset_name}")

        path = os.path.join(
            OUTPUT_DIR,
            f"confusion_{self.dataset_name}.png"
        )

        plt.savefig(path, bbox_inches="tight", dpi=150)
        plt.close()

    def save_decision_boundary(
            self,
            model,
            y_true
    ):
        if self.X_test.shape[1] != 2:
            return

        x = self.X_test.numpy()

        x_min, x_max = (
            x[:, 0].min() - 1,
            x[:, 0].max() + 1
        )

        y_min, y_max = (
            x[:, 1].min() - 1,
            x[:, 1].max() + 1
        )

        step = 0.02

        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, step),
            np.arange(y_min, y_max, step)
        )

        grid = np.c_[
            xx.ravel(),
            yy.ravel()
        ]

        grid_tensor = torch.tensor(
            grid,
            dtype=torch.float32
        )

        with torch.no_grad():
            logits = model(grid_tensor)

            predictions = torch.argmax(
                logits,
                dim=1
            )

        predictions = predictions.numpy().reshape(xx.shape)

        fig, ax = plt.subplots(figsize=(9, 7))

        ax.contourf(
            xx,
            yy,
            predictions,
            alpha=0.35,
            cmap="tab10"
        )

        scatter = ax.scatter(
            x[:, 0],
            x[:, 1],
            c=y_true,
            cmap="tab10",
            edgecolors="black",
            linewidths=0.3,
            s=20
        )

        plt.title(
            f"Decision Boundary: {self.dataset_name}"
        )

        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")

        plt.grid(True)

        unique_classes = np.unique(y_true)

        colors = plt.cm.tab10(
            np.linspace(0, 1, len(unique_classes))
        )

        legend_handles = [
            Patch(
                color=colors[i],
                label=f"Digit {cls}"
            )
            for i, cls in enumerate(unique_classes)
        ]

        ax.legend(
            handles=legend_handles,
            title="Classes",
            loc="upper right"
        )

        path = os.path.join(
            OUTPUT_DIR,
            f"decision_boundary_{self.dataset_name}.png"
        )

        plt.savefig(
            path,
            bbox_inches="tight",
            dpi=150
        )

        plt.close()


    def run(self):

        model = self.load_model()

        y_pred = self.predict(model)

        y_true, y_pred = self.evaluate(y_pred)

        self.save_confusion_matrix(y_true, y_pred)

        self.save_decision_boundary(
            model,
            y_true
        )


def main():

    models = {
        "mnist_flatten": "mnist_flatten.pth",
        "mnist_density_center": "mnist_density_center.pth",
        "mnist_symmetry": "mnist_symmetry.pth",
        "mnist_grid_feature": "mnist_grid_feature.pth",
        "mnist_projection_feature": "mnist_projection_feature.pth",
        "iris": "iris.pth",
        "wine": "wine.pth"
    }

    for dataset_name, model_path in models.items():

        if not os.path.exists("./saved_models/" + model_path):
            print(f"Skipping {dataset_name} (missing model)")
            continue

        Predictor(dataset_name, model_path).run()


if __name__ == "__main__":
    main()