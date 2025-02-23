# sim_cells.py - Simulates cells and saves data
import pandas as pd
import torch
from torch import nn
from torchdiffeq import odeint


class TwoCellSystem(nn.Module):
    def __init__(self):
        super().__init__()
        self.movement = torch.tensor(
            [[0.0, -1.0], [1.0, 0.0]]  # dx/dt = -y  # dy/dt = x
        )

    def forward(self, t, state):
        return self.movement


# Create model
model = TwoCellSystem()

# Initial positions
initial_state = torch.tensor(
    [[-1.0, 0.0], [1.0, 0.0]]  # cell 1 position (x,y)  # cell 2 position (x,y)
)

# Time points (0 to 5 seconds, 50 points for smoother visualization)
t = torch.linspace(0, 5, 50)

# Solve and get positions over time
with torch.no_grad():
    positions = odeint(model, initial_state, t)

# Convert to pandas DataFrame
data = []
for i in range(len(t)):
    data.append(
        {
            "time": t[i].item(),
            "cell1_x": positions[i, 0, 0].item(),
            "cell1_y": positions[i, 0, 1].item(),
            "cell2_x": positions[i, 1, 0].item(),
            "cell2_y": positions[i, 1, 1].item(),
        }
    )

# Save to CSV
df = pd.DataFrame(data)
df.to_csv("cell_trajectories.csv", index=False)
print("Data saved to cell_trajectories.csv")
