import torch
import numpy as np

# Example model (a simple linear model for demonstration)
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = torch.nn.Linear(3, 1)  # Input size 3, output size 1

    def forward(self, x):
        return self.linear(x)

# Instantiate the model
model = SimpleModel()

# Example input
last_seq = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)  # Shape: (1, 3)

# Forward pass
future_scaled = model(last_seq).detach().cpu().numpy()

print(future_scaled)
