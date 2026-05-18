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

            outputs = self.model(x)
            loss = self.criterion(outputs, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            predictions = outputs.argmax(dim=1)
            correct += (predictions == y).sum().item()
            total += y.size(0)

        return total_loss / len(self.train_loader), correct / total

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

        return total_loss / len(self.test_loader), correct / total

    def fit(self, epochs, patience=10):

        history = {
            "train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": []
        }

        best_loss = float("inf")
        best_state = None
        counter = 0

        for epoch in range(1, epochs + 1):

            train_loss, train_acc = self.train_epoch()
            test_loss, test_acc = self.evaluate()

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["test_loss"].append(test_loss)
            history["test_acc"].append(test_acc)

            print(
                f"[Epoch {epoch}/{epochs}] "
                f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
                f"test_loss={test_loss:.4f} | test_acc={test_acc:.4f}"
            )

            if test_loss < best_loss:
                best_loss = test_loss
                best_state = {
                    k: v.cpu().clone()
                    for k, v in self.model.state_dict().items()
                }
                counter = 0
            else:
                counter += 1

            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return history