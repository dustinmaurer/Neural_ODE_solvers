import matplotlib.animation as animation
import matplotlib.pyplot as plt
import pandas as pd


def animate_trajectories(csv_file="cell_trajectories.csv"):
    # Read the data
    df = pd.read_csv(csv_file)

    # Pivot the data to get x, y values for each cell at each time step
    df_pivot = df.pivot(
        index=["time", "cell_id"], columns="coordinate_id", values="value"
    ).reset_index()
    df_pivot.columns = ["time", "cell_id", "x", "y"]

    # Get unique cells
    cell_ids = df_pivot["cell_id"].unique()

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title("Cell Trajectories")
    ax.grid(True)

    # Determine axis limits
    margin = 0.1
    x_min, x_max = df_pivot["x"].min(), df_pivot["x"].max()
    y_min, y_max = df_pivot["y"].min(), df_pivot["y"].max()
    x_range, y_range = x_max - x_min, y_max - y_min
    ax.set_xlim(x_min - margin * x_range, x_max + margin * x_range)
    ax.set_ylim(y_min - margin * y_range, y_max + margin * y_range)

    # Initialize empty plots for paths and positions
    paths = {
        cell_id: ax.plot([], [], label=f"Cell {cell_id}")[0] for cell_id in cell_ids
    }
    points = {cell_id: ax.plot([], [], "o", markersize=8)[0] for cell_id in cell_ids}

    def init():
        for cell_id in cell_ids:
            paths[cell_id].set_data([], [])
            points[cell_id].set_data([], [])
        ax.legend()
        return list(paths.values()) + list(points.values())

    def animate(frame):
        ax.set_title(f"Cell Trajectories (Time {df_pivot['time'].iloc[frame]:.2f})")

        for cell_id in cell_ids:
            df_cell = df_pivot[df_pivot["cell_id"] == cell_id]
            x_data, y_data = (
                df_cell["x"].iloc[: frame + 1],
                df_cell["y"].iloc[: frame + 1],
            )
            paths[cell_id].set_data(x_data, y_data)

            # Update current position
            if frame < len(df_cell):
                x_current, y_current = (
                    df_cell["x"].iloc[frame],
                    df_cell["y"].iloc[frame],
                )
                points[cell_id].set_data(x_current, y_current)

        return list(paths.values()) + list(points.values())

    # Create animation
    frames = df_pivot["time"].nunique()
    ani = animation.FuncAnimation(
        fig, animate, frames=frames, init_func=init, blit=True, interval=50
    )

    # Save animation
    ani.save("cell_trajectories.gif", writer="pillow")
    print("Animation saved as cell_trajectories.gif")

    plt.show()


# Run the animation
if __name__ == "__main__":
    animate_trajectories()
