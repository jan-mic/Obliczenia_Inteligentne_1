import numpy as np
from torchvision import datasets

from extractors.flatten_extractor import FlattenExtractor
from extractors.density_center_extractor import DensityCenterExtractor
from extractors.grid_feature_extractor import GridFeatureExtractor
from extractors.projection_feature_extraction import ProjectionFeatureExtractor
from extractors.symmetry_extractor import SymmetryExtractor


def process_mnist(split="train", extractor=None):

    mnist = datasets.MNIST(
        '../data/.',
        train=(split == "train"),
        download=True
    )

    x = mnist.data.numpy()
    y = mnist.targets.numpy()

    x_features = extractor.fit_transform(x)

    return x_features, y


def save_processed_dataset(name, extractor):

    x_train, y_train = process_mnist(
        split="train",
        extractor=extractor
    )

    np.save(f"../data/processed/{name}_X_train.npy", x_train)
    np.save(f"../data/processed/{name}_y_train.npy", y_train)

    x_test, y_test = process_mnist(
        split="test",
        extractor=extractor
    )

    np.save(f"../data/processed/{name}_X_test.npy", x_test)
    np.save(f"../data/processed/{name}_y_test.npy", y_test)

    print(f"{name} saved successfully")
    print(f"Train shape: {x_train.shape}")
    print(f"Test shape: {x_test.shape}")


def main():

    save_processed_dataset(
        name="mnist_flatten",
        extractor=FlattenExtractor()
    )

    save_processed_dataset(
        name="mnist_density_center",
        extractor=DensityCenterExtractor()
    )

    save_processed_dataset(
        name="mnist_symmetry",
        extractor=SymmetryExtractor()
    )

    save_processed_dataset(
        name="mnist_grid_feature",
        extractor=GridFeatureExtractor()
    )

    save_processed_dataset(
        name="mnist_projection_feature",
        extractor=ProjectionFeatureExtractor()
    )

if __name__ == "__main__":
    main()