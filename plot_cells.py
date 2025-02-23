# plot_cells.py - Visualizes the cell trajectories
import matplotlib.pyplot as plt
import pandas as pd

# Read the data
df = pd.read_csv("cell_trajectories.csv")

# Create the plot
plt.figure(figsize=(10, 10))

# Plot trajectories
plt.plot(df["cell1_x"], df["cell1_y"], "b-", label="Cell 1 path")
plt.plot(df["cell2_x"], df["cell2_y"], "r-", label="Cell 2 path")

# Plot start points
plt.plot(df["cell1_x"].iloc[0], df["cell1_y"].iloc[0], "bo", label="Cell 1 start")
plt.plot(df["cell2_x"].iloc[0], df["cell2_y"].iloc[0], "ro", label="Cell 2 start")

# Plot end points
plt.plot(df["cell1_x"].iloc[-1], df["cell1_y"].iloc[-1], "bx", label="Cell 1 end")
plt.plot(df["cell2_x"].iloc[-1], df["cell2_y"].iloc[-1], "rx", label="Cell 2 end")

plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("Cell Trajectories")
plt.legend()
plt.grid(True)
plt.axis("equal")  # Make axes equal scale

# Save the plot
plt.savefig("cell_trajectories.png")
print("Plot saved as cell_trajectories.png")

# Display the plot (if running in interactive environment)
plt.show()
