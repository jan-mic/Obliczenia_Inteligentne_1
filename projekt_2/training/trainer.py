import os
import torch


class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        dataset_name,
        device="cpu"
    ):

        self.model = model.to(device)

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.criterion = criterion
        self.optimizer = optimizer

        self.dataset_name = dataset_name

        self.device = device

    def train_epoch(self):

        self.model.train()

        total_loss = 0
        correct = 0
        total = 0

        for x, y in self.train_loader:

            x = x.to(self.device)
            y = y.to(self.device)

            # forward
            outputs = self.model(x)

            loss = self.criterion(outputs, y)

            # backward
            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            # stats
            total_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            correct += (predictions == y).sum().item()

            total += y.size(0)

        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def evaluate(self):

        self.model.eval()

        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():

            for x, y in self.test_loader:

                x = x.to(self.device)
                y = y.to(self.device)

                outputs = self.model(x)

                loss = self.criterion(outputs, y)

                total_loss += loss.item()

                predictions = outputs.argmax(dim=1)

                correct += (predictions == y).sum().item()

                total += y.size(0)

        avg_loss = total_loss / len(self.test_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def fit(self, epochs):

        best_acc = 0.0

        history = {
            "train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": []
        }

        os.makedirs("saved_models", exist_ok=True)

        for epoch in range(epochs):

            train_loss, train_acc = self.train_epoch()

            test_loss, test_acc = self.evaluate()

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            history["test_loss"].append(test_loss)
            history["test_acc"].append(test_acc)

            if test_acc > best_acc:

                best_acc = test_acc

                torch.save({
                    "model_state_dict": self.model.state_dict(),
                    "input_dim": self.model.input_dim,
                    "hidden_dim": self.model.hidden_dim,
                    "num_classes": self.model.num_classes,
                    "dataset": self.dataset_name
                }, f"saved_models/{self.dataset_name}.pth")

                print(
                    f"New best model saved! "
                    f"Acc: {best_acc:.4f}"
                )

            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Test Loss: {test_loss:.4f} | "
                f"Test Acc: {test_acc:.4f}"
            )

        return history