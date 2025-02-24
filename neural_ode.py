import pandas as pd
import torch
from torchdiffeq import odeint


# Define TwoCellSystem model
class TwoCellSystem(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Rotation matrix to induce circular motion
        self.movement = torch.tensor(
            [[0.0, -1.0], [1.0, 0.0]]  # dx/dt = -y  # dy/dt = x
        )

    def forward(self, t, state):
        return torch.matmul(state, self.movement.T)


# Create model
model = TwoCellSystem()

# Initial positions for two cells
initial_state = torch.tensor([[-1.0, 0.0], [1.0, 0.0]])  # Cell 1  # Cell 2

# Time points (0 to 5 seconds, 50 points for smoother visualization)
t = torch.linspace(0, 5, 50)

# Solve ODE
with torch.no_grad():
    positions = odeint(model, initial_state, t)

# Reshape the data into long format
data = []
for i in range(len(t)):
    for cell_id in range(2):  # Two cells
        for coord_id in range(2):  # x and y coordinates
            data.append(
                {
                    "time": t[i].item(),
                    "cell_id": cell_id,
                    "coordinate_id": coord_id,  # 0 for x, 1 for y
                    "value": positions[i, cell_id, coord_id].item(),
                }
            )

# Convert to Pandas DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("cell_trajectories.csv", index=False)
print("Data saved to cell_trajectories.csv")
