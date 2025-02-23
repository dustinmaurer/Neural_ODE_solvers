import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def animate_trajectories(csv_file="cell_trajectories.csv"):
    # Read the data
    df = pd.read_csv(csv_file)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))

    def init():
        # Initial plot setup
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_title("Cell Trajectories")
        ax.grid(True)

        # Set axes limits based on data
        margin = 0.1  # 10% margin
        x_min = min(
            df[["cell1_x", "cell2_x"]].min().min(),
            df[["cell1_x", "cell2_x"]].min().min(),
        )
        x_max = max(
            df[["cell1_x", "cell2_x"]].max().max(),
            df[["cell1_x", "cell2_x"]].max().max(),
        )
        y_min = min(
            df[["cell1_y", "cell2_y"]].min().min(),
            df[["cell1_y", "cell2_y"]].min().min(),
        )
        y_max = max(
            df[["cell1_y", "cell2_y"]].max().max(),
            df[["cell1_y", "cell2_y"]].max().max(),
        )

        x_range = x_max - x_min
        y_range = y_max - y_min

        ax.set_xlim(x_min - margin * x_range, x_max + margin * x_range)
        ax.set_ylim(y_min - margin * y_range, y_max + margin * y_range)

        return []

    def animate(frame):
        ax.clear()
        init()

        # Plot trajectories up to current frame
        ax.plot(
            df["cell1_x"].iloc[: frame + 1],
            df["cell1_y"].iloc[: frame + 1],
            "b-",
            alpha=0.5,
            label="Cell 1 path",
        )
        ax.plot(
            df["cell2_x"].iloc[: frame + 1],
            df["cell2_y"].iloc[: frame + 1],
            "r-",
            alpha=0.5,
            label="Cell 2 path",
        )

        # Plot start points
        ax.plot(
            df["cell1_x"].iloc[0], df["cell1_y"].iloc[0], "bo", label="Cell 1 start"
        )
        ax.plot(
            df["cell2_x"].iloc[0], df["cell2_y"].iloc[0], "ro", label="Cell 2 start"
        )

        # Plot current positions
        ax.plot(
            df["cell1_x"].iloc[frame],
            df["cell1_y"].iloc[frame],
            "bo",
            markersize=10,
            alpha=1,
        )
        ax.plot(
            df["cell2_x"].iloc[frame],
            df["cell2_y"].iloc[frame],
            "ro",
            markersize=10,
            alpha=1,
        )

        ax.legend()
        return []

    # Create animation
    frames = len(df)
    ani = animation.FuncAnimation(
        fig, animate, frames=frames, init_func=init, blit=True, interval=50
    )  # 50ms between frames

    # Save animation
    ani.save("cell_trajectories.gif", writer="pillow")
    print("Animation saved as cell_trajectories.gif")

    # Display animation (if in interactive environment)
    plt.show()


# Run the animation
if __name__ == "__main__":
    animate_trajectories()
