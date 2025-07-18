import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.cm import RdBu
from matplotlib.colors import Normalize
from matplotlib.patches import Circle
from matplotlib.widgets import Button

# Initialize figure and axis
fig, ax = plt.subplots()
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_title("Brownian Motion with Force-Inhibited Growth")


# Cell list: start with one circle
def init_cell():
    return {
        "patch": Circle((5, 5), radius=0.5, color="blue", alpha=0.5),
        "velocity": np.array([0.0, 0.0]),
        "age": 0.0,
        "label": ax.text(
            5, 5, "(0.00, 0.50, 0.00)", ha="center", va="center", fontsize=8
        ),
    }


cells = [init_cell()]
for cell in cells:
    ax.add_patch(cell["patch"])

# Animation parameters
noise_sigma = 0.02  # Brownian motion strength
running = False  # Start paused
dt = 0.05  # Time step (seconds)
k_spring = 5  # Spring constant for collisions
selected_cell = None  # Track selected cell
b_drag = 1.0  # Drag coefficient

# Growth parameters (Hill equation based on size)
V_max = 0.3  # Max growth rate (radius units/second)
r_max = 1.0  # Maximum radius
r_min = 0.1  # Minimum radius to prevent vanishing
K = 0.5  # Growth capacity for half-maximal rate
n = 2.0  # Hill coefficient

# Sigmoid inhibition parameters
s = 1.0  # Steepness of sigmoid
F0 = 1.0  # Force magnitude for 50% inhibition
decay = 0.02  # Baseline growth decay

# Color mapping for growth rate
norm = Normalize(vmin=-0.01, vmax=0.01)  # Map growth_rate to colors
cmap = RdBu  # Red (negative), Blue (positive)


# Animation update function
def update(frame):
    global cells, selected_cell
    if running:
        # Handle auto-division based on size
        new_cells = []
        cells_to_keep = []
        for cell in cells:
            r = cell["patch"].radius
            if r >= 0.93 * r_max:  # Within 7% of max radius
                x, y = cell["patch"].center
                cell["label"].remove()

                r_new = r / np.sqrt(2)
                offset = r_new
                angle = np.random.uniform(0, 2 * np.pi)
                dx = offset * np.cos(angle)
                dy = offset * np.sin(angle)

                cell1 = {
                    "patch": Circle(
                        (x + dx, y + dy), radius=r_new, color="blue", alpha=0.5
                    ),
                    "velocity": cell["velocity"] * 0.5,
                    "age": 0.0,
                    "label": ax.text(
                        x + dx,
                        y + dy,
                        f"(0.00, {r_new:.2f}, 0.00)",
                        ha="center",
                        va="center",
                        fontsize=8,
                    ),
                }
                cell2 = {
                    "patch": Circle(
                        (x - dx, y - dy), radius=r_new, color="blue", alpha=0.5
                    ),
                    "velocity": cell["velocity"] * 0.5,
                    "age": 0.0,
                    "label": ax.text(
                        x - dx,
                        y - dy,
                        f"(0.00, {r_new:.2f}, 0.00)",
                        ha="center",
                        va="center",
                        fontsize=8,
                    ),
                }
                ax.add_patch(cell1["patch"])
                ax.add_patch(cell2["patch"])
                new_cells.extend([cell1, cell2])
                if cell == selected_cell:
                    selected_cell = None
            else:
                cells_to_keep.append(cell)

        cells = cells_to_keep + new_cells

        # Compute collisions
        positions = np.array([cell["patch"].center for cell in cells])
        radii = np.array([cell["patch"].radius for cell in cells])
        num_cells = len(cells)

        forces = np.zeros((num_cells, 2))
        for i in range(num_cells):
            for j in range(i + 1, num_cells):
                delta = positions[j] - positions[i]  # From j to i
                dist = np.linalg.norm(delta)
                r_sum = radii[i] + radii[j]
                if dist < r_sum and dist > 0:
                    overlap = r_sum - dist
                    force_dir = delta / dist  # Points from i to j
                    force_mag = k_spring * overlap
                    forces[i] -= force_mag * force_dir  # Push i away from j
                    forces[j] += force_mag * force_dir  # Push j away from i

        # Update each cell
        for i, cell in enumerate(cells):
            x, y = cell["patch"].center
            r = cell["patch"].radius
            vx, vy = cell["velocity"]
            age = cell["age"]

            # Brownian motion
            vx += np.random.normal(0, noise_sigma)
            vy += np.random.normal(0, noise_sigma)

            # Collision force
            vx += forces[i][0] * dt
            vy += forces[i][1] * dt

            # Drag force: F = -b * v
            vx += -b_drag * vx * dt
            vy += -b_drag * vy * dt

            # Update position
            x += vx * dt
            y += vy * dt

            # Bounce off edges
            if x - r < 0:
                x = r
                vx = -vx
            elif x + r > 10:
                x = 10 - r
                vx = -vx
            if y - r < 0:
                y = r
                vy = -vy
            elif y + r > 10:
                y = 10 - r
                vy = -vy

            # Growth with inhibition
            growth_capacity = max(r_max - r, 0)
            growth_rate = V_max * (growth_capacity**n / (K**n + growth_capacity**n))
            # Sigmoid inhibition: 1 / (1 + exp(-s * (|F| - F0)))
            force_mag = np.linalg.norm(forces[i])
            inhibition = 1 / (1 + np.exp(-s * (force_mag - F0)))
            growth_rate *= (
                1 - inhibition - decay
            )  # Reduce growth by inhibition and decay
            r += growth_rate * dt
            r = max(r, r_min)  # Prevent vanishing

            # Update color based on growth_rate
            color = cmap(norm(growth_rate))
            cell["patch"].set_color(color)
            if cell == selected_cell:
                cell["patch"].set_color("yellow")  # Override for selection

            # Update age
            age += dt

            # Update cell
            cell["patch"].center = (x, y)
            cell["patch"].radius = r
            cell["velocity"] = np.array([vx, vy])
            cell["age"] = age
            cell["label"].set_position((x, y))
            cell["label"].set_text(f"({age:.2f}, {r:.2f}, {growth_rate:.2f})")

    return [cell["patch"] for cell in cells] + [cell["label"] for cell in cells]


# Toggle animation state
def toggle(event):
    global running
    running = not running
    toggle_button.label.set_text("Resume" if not running else "Pause")


# Divide selected cell
def divide(event):
    global selected_cell
    # If no cell selected, pick closest to plot center (5, 5)
    if not selected_cell and cells:
        ref_x, ref_y = 5, 5  # Plot center
        min_dist = float("inf")
        closest_cell = None
        for cell in cells:
            x, y = cell["patch"].center
            dist = np.sqrt((ref_x - x) ** 2 + (ref_y - y) ** 2)
            if dist < min_dist:
                min_dist = dist
                closest_cell = cell
        selected_cell = closest_cell
        update_selection()

    if selected_cell:
        cells.remove(selected_cell)
        x, y = selected_cell["patch"].center
        r = selected_cell["patch"].radius
        selected_cell["label"].remove()

        r_new = r / np.sqrt(2)
        offset = r_new
        angle = np.random.uniform(0, 2 * np.pi)
        dx = offset * np.cos(angle)
        dy = offset * np.sin(angle)

        cell1 = {
            "patch": Circle((x + dx, y + dy), radius=r_new, color="blue", alpha=0.5),
            "velocity": selected_cell["velocity"] * 0.5,
            "age": 0.0,
            "label": ax.text(
                x + dx,
                y + dy,
                f"(0.00, {r_new:.2f}, 0.00)",
                ha="center",
                va="center",
                fontsize=8,
            ),
        }
        cell2 = {
            "patch": Circle((x - dx, y - dy), radius=r_new, color="blue", alpha=0.5),
            "velocity": selected_cell["velocity"] * 0.5,
            "age": 0.0,
            "label": ax.text(
                x - dx,
                y - dy,
                f"(0.00, {r_new:.2f}, 0.00)",
                ha="center",
                va="center",
                fontsize=8,
            ),
        }

        ax.add_patch(cell1["patch"])
        ax.add_patch(cell2["patch"])
        cells.extend([cell1, cell2])
        selected_cell = None
        update_selection()


# Reset to one cell
def reset(event):
    global cells, selected_cell
    selected_cell = None
    for cell in cells:
        cell["patch"].remove()
        cell["label"].remove()
    cells = [init_cell()]
    ax.add_patch(cells[0]["patch"])
    update_selection()


# Handle cell selection
def on_click(event):
    global selected_cell
    if event.inaxes != ax:
        return
    mouse_x, mouse_y = event.xdata, event.ydata
    if mouse_x is None or mouse_y is None:
        return

    min_dist = float("inf")
    closest_cell = None
    for cell in cells:
        x, y = cell["patch"].center
        r = cell["patch"].radius
        dist = np.sqrt((mouse_x - x) ** 2 + (mouse_y - y) ** 2)
        if dist < r and dist < min_dist:
            min_dist = dist
            closest_cell = cell

    selected_cell = closest_cell
    update_selection()


# Update cell colors based on selection
def update_selection():
    for cell in cells:
        if cell == selected_cell:
            cell["patch"].set_color("yellow")
        else:
            # Restore color based on growth_rate
            r = cell["patch"].radius
            growth_capacity = max(r_max - r, 0)
            growth_rate = V_max * (growth_capacity**n / (K**n + growth_capacity**n))
            force_mag = np.linalg.norm(cell.get("last_force", np.array([0.0, 0.0])))
            inhibition = 1.2 / (1 + np.exp(-s * (force_mag - F0)))
            growth_rate *= 1 - inhibition - decay
            color = cmap(norm(growth_rate))
            cell["patch"].set_color(color)


# Connect click handler
fig.canvas.mpl_connect("button_press_event", on_click)

# Create buttons
toggle_ax = plt.axes([0.65, 0.025, 0.1, 0.04])
toggle_button = Button(toggle_ax, "Resume")
toggle_button.on_clicked(toggle)

divide_ax = plt.axes([0.775, 0.025, 0.1, 0.04])
divide_button = Button(divide_ax, "Divide")
divide_button.on_clicked(divide)

reset_ax = plt.axes([0.9, 0.025, 0.1, 0.04])
reset_button = Button(reset_ax, "Reset")
reset_button.on_clicked(reset)

# Setup animation
ani = FuncAnimation(fig, update, interval=50, blit=True, cache_frame_data=False)

# Show window
plt.show()
