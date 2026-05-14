import os

import torch
import torch.nn as nn
import torch.optim as optim

from datasets.loaders import load_dataset
from models.mlp import MLP
from training.trainer import Trainer


def main(dataset_name="mnist_flatten"):
    print("Starting for: " + dataset_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset, test_dataset = load_dataset(dataset_name)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False
    )

    sample_x, _ = train_dataset[0]
    input_dim = sample_x.shape[0]

    num_classes = len(set(train_dataset.y.numpy())) if hasattr(train_dataset, "y") else 10

    if dataset_name.startswith("mnist"):
        num_classes = 10

    model = MLP(
        input_dim=input_dim,
        hidden_dim=128,
        num_classes=num_classes,
        use_relu=True
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=0.01
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        dataset_name=dataset_name
    )

    trainer.fit(epochs=50)


if __name__ == "__main__":

    main("iris")
    main("wine")
    main("mnist_flatten")

    main("mnist_density_center")
    main("mnist_symmetry")

    main("mnist_grid_feature")
    main("mnist_projection_feature")
