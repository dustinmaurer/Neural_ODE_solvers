import torch
import torch.nn as nn
from torchdiffeq import odeint


class CellLifecycleODE(nn.Module):
    def __init__(self):
        super().__init__()
        # Each cell state now includes:
        # - x, y, z position (3)
        # - viability score (1) - represents cell health/death probability
        # - proliferation potential (1) - likelihood of division
        self.state_dim = 5  # 3 spatial + 2 lifecycle dimensions

        # Neural network for cell dynamics
        self.dynamics_net = nn.Sequential(
            nn.Linear(self.state_dim, 32), nn.Tanh(), nn.Linear(32, self.state_dim)
        )

    def handle_lifecycle_events(self, state, threshold=0.1):
        """
        Handle discrete events (death and division) based on continuous state
        Returns: Updated state tensor with removed/added cells
        """
        # Reshape to get individual cell states
        cells = state.view(-1, self.state_dim)

        # Get viability and proliferation scores
        viability = cells[:, 3]
        proliferation = cells[:, 4]

        # Remove cells with low viability (apoptosis)
        viable_mask = viability > threshold
        surviving_cells = cells[viable_mask]

        # Handle cell division
        dividing_mask = proliferation > 0.8  # High proliferation score
        dividing_cells = surviving_cells[dividing_mask]

        if len(dividing_cells) > 0:
            new_cells = []
            for cell in dividing_cells:
                # Create two daughter cells with slightly perturbed positions
                daughter1 = cell.clone()
                daughter2 = cell.clone()

                # Add small random offset to positions
                offset = torch.randn(3) * 0.1
                daughter1[:3] += offset
                daughter2[:3] -= offset

                # Reset proliferation potential
                daughter1[4] = 0.2  # Give time before next division
                daughter2[4] = 0.2

                new_cells.extend([daughter1, daughter2])

            # Add new cells to surviving population
            if new_cells:
                new_cells_tensor = torch.stack(new_cells)
                surviving_cells = torch.cat([surviving_cells, new_cells_tensor])

        return surviving_cells

    def forward(self, t, state):
        """
        Compute continuous state changes and handle discrete events
        """
        # Continuous dynamics
        derivatives = self.dynamics_net(state)

        # Modify derivatives based on local environment
        positions = state[:, :3]
        densities = self.compute_local_density(positions)

        # Adjust viability based on density (overcrowding leads to death)
        derivatives[:, 3] -= 0.1 * densities

        # Adjust proliferation based on density (contact inhibition)
        derivatives[:, 4] -= 0.2 * densities

        return derivatives

    def compute_local_density(self, positions, radius=1.0):
        """
        Compute local cell density for each cell
        """
        # Compute pairwise distances
        diffs = positions.unsqueeze(0) - positions.unsqueeze(1)
        distances = torch.norm(diffs, dim=-1)

        # Count neighbors within radius
        density = (distances < radius).float().sum(dim=-1)
        return density


# Example usage
def simulate_with_lifecycle(model, initial_state, times):
    trajectories = []
    current_state = initial_state

    for t1, t2 in zip(times[:-1], times[1:]):
        # Solve ODE for short time interval
        trajectory = odeint(model, current_state, torch.tensor([t1, t2]))

        # Handle discrete events
        current_state = model.handle_lifecycle_events(trajectory[-1])

        trajectories.append(trajectory)

    return trajectories


# Initialize
model = CellLifecycleODE()
initial_cells = torch.randn(10, 5)  # 10 cells with random initial states
initial_cells[:, 3] = 0.9  # Set initial viability high
initial_cells[:, 4] = 0.2  # Set initial proliferation potential low
