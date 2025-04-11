import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from matplotlib.widgets import Button

# Initialize figure and axis
fig, ax = plt.subplots()
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_title("Brownian Motion with Size-Based Growth and Division")

# Cell list: start with one circle at (5, 5), radius 0.5
cells = [
    {
        "patch": Circle((5, 5), radius=0.5, color="blue", alpha=0.5),
        "velocity": np.array([0.0, 0.0]),
        "age": 0.0,
        "label": ax.text(5, 5, "(0.00, 0.50)", ha="center", va="center", fontsize=8),
    }
]
for cell in cells:
    ax.add_patch(cell["patch"])

# Animation parameters
noise_sigma = 0.1  # Strength of Brownian motion
running = True  # Animation state
dt = 0.05  # Time step (seconds per frame)

# Growth parameters (Hill equation based on size)
V_max = 0.1  # Max growth rate (radius units per second)
r_max = 1.0  # Maximum radius
K = 0.5  # Growth capacity (r_max - r) when growth rate is half-maximal
n = 2.0  # Hill coefficient


# Animation update function
def update(frame):
    if running:
        for cell in cells:
            # Get current state
            x, y = cell["patch"].center
            r = cell["patch"].radius
            vx, vy = cell["velocity"]
            age = cell["age"]

            # Apply Brownian motion
            vx += np.random.normal(0, noise_sigma)
            vy += np.random.normal(0, noise_sigma)

            # Update position
            x += vx * dt
            y += vy * dt

            # Bounce off edges
            if x - r < 0:  # Left edge
                x = r
                vx = -vx
            elif x + r > 10:  # Right edge
                x = 10 - r
                vx = -vx
            if y - r < 0:  # Bottom edge
                y = r
                vy = -vy
            elif y + r > 10:  # Top edge
                y = 10 - r
                vy = -vy

            # Growth (Hill: dr/dt = V_max * ((r_max - r)^n / (K^n + (r_max - r)^n)))
            growth_capacity = max(r_max - r, 0)  # Prevent negative capacity
            growth_rate = V_max * (growth_capacity**n / (K**n + growth_capacity**n))
            r += growth_rate * dt
            r = min(r, r_max)  # Cap at r_max

            # Update age
            age += dt

            # Update cell
            cell["patch"].center = (x, y)
            cell["patch"].radius = r
            cell["velocity"] = np.array([vx, vy])
            cell["age"] = age
            # Update label: (age, radius)
            cell["label"].set_position((x, y))
            cell["label"].set_text(f"({age:.2f}, {r:.2f})")

    return [cell["patch"] for cell in cells] + [cell["label"] for cell in cells]


# Toggle animation state
def toggle(event):
    global running
    running = not running
    toggle_button.label.set_text("Resume" if not running else "Pause")


# Divide first cell
def divide(event):
    global cells
    if cells:
        mother = cells.pop(0)
        x, y = mother["patch"].center
        r = mother["patch"].radius
        mother["label"].remove()  # Remove mother’s label

        # New radius (half the area: pi*r_new^2 = pi*r^2/2)
        r_new = r / np.sqrt(2)

        # Place new cells, contact edge at mother’s centroid
        offset = r_new
        angle = np.random.uniform(0, 2 * np.pi)
        dx = offset * np.cos(angle)
        dy = offset * np.sin(angle)

        # Create two new cells
        cell1 = {
            "patch": Circle((x + dx, y + dy), radius=r_new, color="blue", alpha=0.5),
            "velocity": mother["velocity"] * 0.5,
            "age": 0.0,
            "label": ax.text(
                x + dx,
                y + dy,
                f"(0.00, {r_new:.2f})",
                ha="center",
                va="center",
                fontsize=8,
            ),
        }
        cell2 = {
            "patch": Circle((x - dx, y - dy), radius=r_new, color="blue", alpha=0.5),
            "velocity": mother["velocity"] * 0.5,
            "age": 0.0,
            "label": ax.text(
                x - dx,
                y - dy,
                f"(0.00, {r_new:.2f})",
                ha="center",
                va="center",
                fontsize=8,
            ),
        }

        # Add to plot and cells list
        ax.add_patch(cell1["patch"])
        ax.add_patch(cell2["patch"])
        cells.extend([cell1, cell2])


# Create buttons
toggle_ax = plt.axes([0.7, 0.025, 0.1, 0.04])
toggle_button = Button(toggle_ax, "Pause")
toggle_button.on_clicked(toggle)

divide_ax = plt.axes([0.85, 0.025, 0.1, 0.04])
divide_button = Button(divide_ax, "Divide")
divide_button.on_clicked(divide)

# Setup animation
ani = FuncAnimation(fig, update, interval=50, blit=True, cache_frame_data=False)

# Show window
plt.show()
