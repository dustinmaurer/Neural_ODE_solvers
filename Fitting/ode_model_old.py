import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class SimplifiedODEModel(nn.Module):
    def __init__(self, n_trajectories):
        super(SimplifiedODEModel, self).__init__()
        self.shape_weights = nn.Parameter(torch.zeros(n_trajectories, n_trajectories))

    def sigmoid(self, x):
        return torch.sigmoid(x)

    def forward(self, t, y, trajectories, time_points):
        t_idx = torch.searchsorted(time_points, t)
        if t_idx >= len(time_points):
            t_idx = len(time_points) - 1

        traj_at_t = trajectories[:, t_idx]
        shape = self.shape_weights @ traj_at_t
        dydt = self.sigmoid(shape) - y
        return dydt


def rk4_step(model, t, y, dt, y_true, t_tensor):
    k1 = model(t, y, y_true, t_tensor)
    k2 = model(t + dt / 2, y + (dt / 2) * k1, y_true, t_tensor)
    k3 = model(t + dt / 2, y + (dt / 2) * k2, y_true, t_tensor)
    k4 = model(t + dt, y + dt * k3, y_true, t_tensor)
    return y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def fit_simplified_ode_with_epochs(
    trajectories,
    time_points,
    n_epochs=1000,
    lr=0.01,
    tol=1e-5,
    patience=10,
    save_interval=100,
    verbose=False,
):
    print("Starting model fitting with epoch tracking...")
    n_trajectories = trajectories.shape[0]

    trajectory_mins = trajectories.min(axis=1, keepdims=True)
    trajectory_maxs = trajectories.max(axis=1, keepdims=True)
    normalized_trajectories = (trajectories - trajectory_mins) / (
        trajectory_maxs - trajectory_mins + 1e-8
    )

    model = SimplifiedODEModel(n_trajectories)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    t_tensor = torch.tensor(time_points, dtype=torch.float32)
    y_true = torch.tensor(normalized_trajectories, dtype=torch.float32)

    with torch.no_grad():
        model.shape_weights.data.uniform_(-0.1, 0.1)
        print(
            f"Initial weights mean: {model.shape_weights.mean().item():.6f}, std: {model.shape_weights.std().item():.6f}"
        )

    prev_loss = float("inf")
    patience_counter = 0

    epoch_trajectories = {}
    epoch_weights = {}

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        y0 = y_true[:, 0]
        y_pred = [y0]
        dt = t_tensor[1] - t_tensor[0]

        for i in range(len(t_tensor) - 1):
            t = t_tensor[i]
            y_current = y_pred[i]
            next_y = rk4_step(model, t, y_current, dt, y_true, t_tensor)
            y_pred.append(next_y)
        y_pred = torch.stack(y_pred, dim=1)

        loss = torch.mean((y_pred - y_true) ** 2)

        if torch.isnan(loss):
            print(f"Epoch {epoch}: NaN detected in loss, stopping early")
            break

        loss.backward()

        # Check gradients
        grad_norm = torch.norm(model.shape_weights.grad)
        if epoch % save_interval == 0:
            print(f"Epoch {epoch}: Gradient norm: {grad_norm.item():.6f}")

        optimizer.step()

        # Verify weights are changing (only compare when we have a previous saved epoch)
        weights_changed = True  # Default to True for early epochs
        if epoch >= save_interval and (epoch - save_interval) in epoch_weights:
            prev_weights = torch.tensor(
                epoch_weights[epoch - save_interval], dtype=torch.float32
            )
            weights_changed = not torch.allclose(
                model.shape_weights.data, prev_weights, atol=1e-6
            )

        loss_val = loss.item()
        loss_change = abs(prev_loss - loss_val)
        if loss_change < tol:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"Epoch {epoch}: Loss change ({loss_change:.6f}) below tolerance ({tol}) "
                    f"for {patience} epochs, stopping early"
                )
                denormalized_y_pred = (
                    y_pred.detach().numpy() * (trajectory_maxs - trajectory_mins)
                    + trajectory_mins
                )
                epoch_trajectories[epoch] = denormalized_y_pred
                epoch_weights[epoch] = model.shape_weights.detach().numpy().copy()
                break
        else:
            patience_counter = 0
        prev_loss = loss_val

        if epoch % save_interval == 0 or epoch == n_epochs - 1:
            denormalized_y_pred = (
                y_pred.detach().numpy() * (trajectory_maxs - trajectory_mins)
                + trajectory_mins
            )
            epoch_trajectories[epoch] = denormalized_y_pred
            epoch_weights[epoch] = model.shape_weights.detach().numpy().copy()
            if verbose:
                print(
                    f"Epoch {epoch}, Loss: {loss_val:.6f}, Loss Change: {loss_change:.6f}, "
                    f"Weights Changed: {weights_changed}, "
                    f"Weights Mean: {model.shape_weights.mean().item():.6f}, "
                    f"Weights Std: {model.shape_weights.std().item():.6f}"
                )

    final_y_pred = (
        y_pred.detach().numpy() * (trajectory_maxs - trajectory_mins) + trajectory_mins
    )
    print("Model fitting complete!")
    print(
        f"Final weights mean: {model.shape_weights.mean().item():.6f}, std: {model.shape_weights.std().item():.6f}"
    )

    return (
        model,
        final_y_pred,
        normalized_trajectories,
        epoch_trajectories,
        epoch_weights,
    )


def main(input_path):
    print("Starting main function...")
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    time_points = df["t"].values
    trajectories = df[[col for col in df.columns if col != "t"]].values.T
    print(
        f"Loaded data with {len(time_points)} time points and {trajectories.shape[0]} trajectories"
    )

    save_interval = 10
    (
        model,
        solved_trajectories,
        normalized_trajectories,
        epoch_trajectories,
        epoch_weights,
    ) = fit_simplified_ode_with_epochs(
        trajectories, time_points, save_interval=save_interval, lr=0.001
    )

    # Save final parameters
    shape_weights = model.shape_weights.detach().numpy()
    rows, cols = shape_weights.shape
    params_rows = []
    for i in range(rows):
        for j in range(cols):
            params_rows.append({"row": i, "column": j, "value": shape_weights[i, j]})
    params_df = pd.DataFrame(params_rows)

    # Save weights at each epoch
    weights_rows = []
    for epoch, weights in epoch_weights.items():
        for i in range(weights.shape[0]):
            for j in range(weights.shape[1]):
                weights_rows.append(
                    {"epoch": epoch, "row": i, "column": j, "value": weights[i, j]}
                )
    weights_df = pd.DataFrame(weights_rows)

    # Save solved trajectories
    solved_df = pd.DataFrame({"t": time_points})
    norm_df = pd.DataFrame({"t": time_points})
    for i in range(solved_trajectories.shape[0]):
        solved_df[f"traj_{i}"] = solved_trajectories[i]
        norm_df[f"traj_{i}"] = normalized_trajectories[i]

    # Save epoch trajectories
    epoch_traj_rows = []
    for epoch, epoch_data in epoch_trajectories.items():
        for time_idx, t in enumerate(time_points):
            for traj_idx in range(epoch_data.shape[0]):
                epoch_traj_rows.append(
                    {
                        "epoch": epoch,
                        "t": t,
                        "trajectory": traj_idx,
                        "value": epoch_data[traj_idx, time_idx],
                    }
                )
    epoch_traj_df = pd.DataFrame(epoch_traj_rows)

    output_dir = input_path.parent
    params_path = output_dir / "simplified_fitted_parameters.csv"
    solved_path = output_dir / "simplified_solved_trajectories.csv"
    norm_path = output_dir / "normalized_trajectories.csv"
    epoch_path = output_dir / "epoch_trajectories.csv"
    weights_path = output_dir / "epoch_weights.csv"

    print("Saving output files...")
    params_df.to_csv(params_path, index=False)
    solved_df.to_csv(solved_path, index=False)
    norm_df.to_csv(norm_path, index=False)
    epoch_traj_df.to_csv(epoch_path, index=False)
    weights_df.to_csv(weights_path, index=False)

    print(f"Parameters saved to '{params_path}'")
    print(f"Solved trajectories saved to '{solved_path}'")
    print(f"Normalized trajectories saved to '{norm_path}'")
    print(f"Epoch trajectories saved to '{epoch_path}'")
    print(f"Epoch weights saved to '{weights_path}'")
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit simplified ODE parameters to trajectory data"
    )
    parser.add_argument("input_csv", type=str, help="Path to input CSV file")
    args = parser.parse_args()
    main(args.input_csv)
