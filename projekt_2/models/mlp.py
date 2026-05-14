import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_classes,
        use_relu=True
    ):
        super().__init__()

        self.use_relu = use_relu
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):

        x = self.fc1(x)

        if self.use_relu:
            x = F.relu(x)

        x = self.fc2(x)

        return x