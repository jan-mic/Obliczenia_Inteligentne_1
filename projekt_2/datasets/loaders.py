import numpy as np
from sklearn.datasets import load_wine, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datasets.vector_dataset import VectorDataset

def load_iris_dataset():
    data = load_iris()

    x_train, x_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=264321
    )

    scaler = StandardScaler()

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return (
        VectorDataset(x_train, y_train),
        VectorDataset(x_test, y_test)
    )

def load_wine_dataset():
    data = load_wine()

    x_train, x_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=264321
    )

    scaler = StandardScaler()

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return (
        VectorDataset(x_train, y_train),
        VectorDataset(x_test, y_test)
    )

def load_mnist_flatten_dataset():

    x_train = np.load("./data/processed/mnist_flatten_X_train.npy")
    y_train = np.load("./data/processed/mnist_flatten_y_train.npy")

    x_test = np.load("./data/processed/mnist_flatten_X_test.npy")
    y_test = np.load("./data/processed/mnist_flatten_y_test.npy")

    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    return (
        VectorDataset(x_train, y_train),
        VectorDataset(x_test, y_test)
    )

def load_mnist_density_center_dataset():

    x_train = np.load("./data/processed/mnist_density_center_X_train.npy")
    y_train = np.load("./data/processed/mnist_density_center_y_train.npy")

    x_test = np.load("./data/processed/mnist_density_center_X_test.npy")
    y_test = np.load("./data/processed/mnist_density_center_y_test.npy")

    scaler = StandardScaler()

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return (
        VectorDataset(x_train, y_train),
        VectorDataset(x_test, y_test)
    )

def load_mnist_symmetry_dataset():

    x_train = np.load("./data/processed/mnist_symmetry_X_train.npy")
    y_train = np.load("./data/processed/mnist_symmetry_y_train.npy")

    x_test = np.load("./data/processed/mnist_symmetry_X_test.npy")
    y_test = np.load("./data/processed/mnist_symmetry_y_test.npy")

    scaler = StandardScaler()

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return (
        VectorDataset(x_train, y_train),
        VectorDataset(x_test, y_test)
    )

def load_mnist_grid_feature_dataset():

    x_train = np.load("./data/processed/mnist_grid_feature_X_train.npy")
    y_train = np.load("./data/processed/mnist_grid_feature_y_train.npy")

    x_test = np.load("./data/processed/mnist_grid_feature_X_test.npy")
    y_test = np.load("./data/processed/mnist_grid_feature_y_test.npy")

    scaler = StandardScaler()

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return (
        VectorDataset(x_train, y_train),
        VectorDataset(x_test, y_test)
    )

def load_mnist_projection_feature_dataset():

    x_train = np.load("./data/processed/mnist_projection_feature_X_train.npy")
    y_train = np.load("./data/processed/mnist_projection_feature_y_train.npy")

    x_test = np.load("./data/processed/mnist_projection_feature_X_test.npy")
    y_test = np.load("./data/processed/mnist_projection_feature_y_test.npy")

    scaler = StandardScaler()

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return (
        VectorDataset(x_train, y_train),
        VectorDataset(x_test, y_test)
    )

def load_dataset(name):

    match name:
        case "iris":
            return load_iris_dataset()

        case "wine":
            return load_wine_dataset()

        case "mnist_flatten":
            return load_mnist_flatten_dataset()

        case "mnist_density_center":
            return load_mnist_density_center_dataset()

        case "mnist_symmetry":
            return load_mnist_symmetry_dataset()

        case "mnist_grid_feature":
            return load_mnist_grid_feature_dataset()

        case "mnist_projection_feature":
            return load_mnist_projection_feature_dataset()

        case _:
            raise ValueError(f"Unknown dataset: {name}")