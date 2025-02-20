from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from torchdiffeq import odeint


class EmbryoNeuralODE(nn.Module):
    def __init__(self, n_cells: int = 100, hidden_dim: int = 64):
        super().__init__()
        # State dimension: x, y, z coordinates + additional features per cell
        self.state_dim = 3
        self.n_cells = n_cells

        # Neural network for computing derivatives
        self.net = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.state_dim),
        )

        # Optional: Add spatial awareness
        self.spatial_net = nn.Sequential(
            nn.Linear(
                self.state_dim * 2, hidden_dim
            ),  # Input: cell state + relative positions
            nn.Tanh(),
            nn.Linear(hidden_dim, self.state_dim),
        )

    def compute_cell_interactions(self, state):
        """Compute cell-cell interactions based on spatial relationships"""
        batch_size = state.shape[0]
        # Reshape state to (batch, n_cells, features)
        state = state.view(batch_size, self.n_cells, -1)

        # Compute pairwise distances and interactions
        diff = state.unsqueeze(2) - state.unsqueeze(
            1
        )  # shape: (batch, n_cells, n_cells, features)
        dist = torch.norm(diff, dim=-1, keepdim=True)

        # Simple interaction model: cells influence each other based on distance
        interaction = torch.sum(
            diff / (dist + 1e-8), dim=2
        )  # shape: (batch, n_cells, features)

        return interaction.view(batch_size, -1)

    def forward(self, t, state):
        """
        Compute the derivative of the state with respect to time
        Args:
            t: Current time point
            state: Current state tensor of shape (batch_size, n_cells * state_dim)
        """
        batch_size = state.shape[0]

        # Individual cell dynamics
        individual_dynamics = self.net(state.view(-1, self.state_dim))

        # Cell-cell interactions
        interactions = self.compute_cell_interactions(state)
        interaction_effects = self.spatial_net(torch.cat([state, interactions], dim=-1))

        # Combine individual dynamics and interactions
        derivative = individual_dynamics + interaction_effects

        # Add some biological constraints (optional)
        # For example, keep cells within a certain volume
        state_reshaped = state.view(batch_size, self.n_cells, -1)
        center_of_mass = torch.mean(state_reshaped, dim=1, keepdim=True)
        displacement = state_reshaped - center_of_mass
        volume_constraint = -0.1 * displacement.view(batch_size, -1)

        return derivative + volume_constraint


class SimulationConfig(BaseModel):
    n_cells: int
    time_points: List[float]
    initial_state: List[List[float]]


app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model = None


@app.on_event("startup")
async def startup_event():
    global model
    model = EmbryoNeuralODE(n_cells=100)
    # Here you would typically load pre-trained weights
    # model.load_state_dict(torch.load("pretrained_weights.pth"))


@app.post("/simulate")
async def simulate_development(config: SimulationConfig):
    # Convert input data to tensors
    initial_state = torch.tensor(config.initial_state, dtype=torch.float32)
    time_points = torch.tensor(config.time_points, dtype=torch.float32)

    # Run ODE solver
    with torch.no_grad():
        trajectory = odeint(
            model,
            initial_state,
            time_points,
            method="dopri5",  # Adaptive step-size solver
            rtol=1e-3,
            atol=1e-3,
        )

    # Convert trajectory to list for JSON serialization
    trajectory_list = trajectory.numpy().tolist()

    return {"trajectory": trajectory_list}


@app.get("/healthcheck")
async def healthcheck():
    return {"status": "healthy"}


def train_model(model, train_data, epochs=100):
    """
    Train the Neural ODE model using real or simulated data
    Args:
        model: EmbryoNeuralODE instance
        train_data: Dictionary containing:
            - initial_states: (batch_size, n_cells * state_dim)
            - time_points: (n_times,)
            - target_trajectories: (n_times, batch_size, n_cells * state_dim)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward pass through ODE solver
        pred_trajectory = odeint(
            model,
            train_data["initial_states"],
            train_data["time_points"],
            method="dopri5",
        )

        # Compute loss
        loss = torch.mean((pred_trajectory - train_data["target_trajectories"]) ** 2)

        # Backward pass
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")


if __name__ == "__main__":
    # Run the FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000)
