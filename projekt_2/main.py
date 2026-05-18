import os
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets.loaders import load_dataset
from models.mlp import MLP
from training.trainer import Trainer


def main(dataset_name="mnist_flatten"):

    print(f"\nStarting grid search for: {dataset_name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("saved_models", exist_ok=True)

    train_dataset, test_dataset = load_dataset(dataset_name)
    # opcje pod gpu
    train_loader = DataLoader(
        train_dataset,
        batch_size=1024,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    # cpu
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False
    )

    sample_x, _ = train_dataset[0]
    input_dim = sample_x.shape[0]

    num_classes = (
        len(set(train_dataset.y.numpy()))
        if hasattr(train_dataset, "y")
        else 10
    )

    if dataset_name.startswith("mnist"):
        num_classes = 10

    hidden_dims = [64, 128, 256, 512]
    learning_rates = [0.1, 0.01, 0.001]
    epoch_values = [150]

    best_accuracy = 0.0
    best_config = None

    for hidden_dim, lr, epochs in itertools.product(
        hidden_dims,
        learning_rates,
        epoch_values
    ):

        print("\n====================================")
        print(f"hidden_dim={hidden_dim}, lr={lr}, epochs={epochs}, dataset_name={dataset_name}" )
        print("====================================")

        model = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            use_relu=True
        ).to(device)

        criterion = nn.CrossEntropyLoss()

        optimizer = optim.SGD(
            model.parameters(),
            lr=lr
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
        history = trainer.fit(epochs=epochs)

        final_acc = history["test_acc"][-1]

        if final_acc > best_accuracy:
            best_accuracy = final_acc

            best_config = {
                "hidden_dim": hidden_dim,
                "learning_rate": lr,
                "epochs": epochs
            }

            torch.save({
                "model_state_dict": model.state_dict(),
                "input_dim": input_dim,
                "hidden_dim": hidden_dim,
                "num_classes": num_classes,
                "dataset": dataset_name,
                "best_accuracy": best_accuracy,
                "best_config": best_config
            }, f"saved_models/{dataset_name}.pth")

            print("\n====================================")
            print("\nModel saved, best for now")
            print("\n====================================")

    print("\n====================================")
    print("BEST CONFIGURATION FOR: " + dataset_name)
    print("====================================")

    print(best_config)
    print(f"Best accuracy: {best_accuracy:.4f}")


if __name__ == "__main__":

    # main("iris")
    # main("wine")
    # main("mnist_flatten")
    # main("mnist_density_center")
    # main("mnist_symmetry")
    main("mnist_grid_feature")
    # main("mnist_projection_feature")