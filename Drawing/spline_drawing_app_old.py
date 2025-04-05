import tkinter as tk
from tkinter import ttk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.interpolate import splev, splprep


class SplineDrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Spline Drawing App")
        self.root.geometry("800x600")

        # Control panel
        self.control_frame = ttk.Frame(root)
        self.control_frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(self.control_frame, text="Sampling points:").pack(
            side=tk.LEFT, padx=5, pady=5
        )
        self.samples_var = tk.StringVar(value="50")
        ttk.Entry(self.control_frame, textvariable=self.samples_var, width=5).pack(
            side=tk.LEFT, padx=5, pady=5
        )

        ttk.Button(self.control_frame, text="Clear", command=self.clear_canvas).pack(
            side=tk.LEFT, padx=5, pady=5
        )
        ttk.Button(
            self.control_frame,
            text="Generate Time Series",
            command=self.generate_time_series,
        ).pack(side=tk.LEFT, padx=5, pady=5)

        # Canvas for drawing
        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Initialize plot
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 100)
        self.ax.set_title("Click and drag to create control points")
        self.ax.grid(True)

        # Data storage
        self.control_points = []
        (self.line,) = self.ax.plot([], [], "ro-")  # Control points
        (self.spline_line,) = self.ax.plot([], [], "b-")  # Spline curve

        # Connect events
        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.canvas.mpl_connect("motion_notify_event", self.on_move)
        self.canvas.mpl_connect("button_release_event", self.on_release)

        # Status variables
        self.dragging = False
        self.selected_point = None

        # Results display
        self.result_frame = ttk.Frame(root)
        self.result_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.result_text = tk.Text(self.result_frame, height=10, width=80)
        self.result_text.pack(fill=tk.X, padx=5, pady=5)

    def on_click(self, event):
        if event.xdata is None or event.ydata is None:
            return

        # Check if clicking near an existing point
        for i, (x, y) in enumerate(self.control_points):
            if abs(x - event.xdata) < 3 and abs(y - event.ydata) < 3:
                self.dragging = True
                self.selected_point = i
                break
        else:
            # If not near any point, add a new one
            self.control_points.append((event.xdata, event.ydata))
            self.selected_point = len(self.control_points) - 1
            self.dragging = True

        self.update_plot()

    def on_move(self, event):
        if self.dragging and event.xdata is not None and event.ydata is not None:
            self.control_points[self.selected_point] = (event.xdata, event.ydata)
            self.update_plot()

    def on_release(self, event):
        self.dragging = False
        self.selected_point = None

    def update_plot(self):
        if len(self.control_points) > 0:
            x, y = zip(*self.control_points)
            self.line.set_data(x, y)

            # Update spline if we have enough points
            if len(self.control_points) >= 2:
                self.update_spline()

        self.canvas.draw()

    def update_spline(self):
        x, y = zip(*sorted(self.control_points))  # Sort by x values

        # If we only have 2 points, just draw a line
        if len(x) == 2:
            self.spline_line.set_data(x, y)
            return

        # Fit spline
        try:
            tck, u = splprep([x, y], s=0)
            u_new = np.linspace(0, 1, 100)
            x_new, y_new = splev(u_new, tck)

            self.spline_line.set_data(x_new, y_new)
        except Exception as e:
            print(f"Error fitting spline: {e}")

    def clear_canvas(self):
        self.control_points = []
        self.line.set_data([], [])
        self.spline_line.set_data([], [])
        self.result_text.delete(1.0, tk.END)
        self.canvas.draw()

    def generate_time_series(self):
        if len(self.control_points) < 2:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(
                tk.END, "Need at least 2 points to generate time series"
            )
            return

        try:
            num_samples = int(self.samples_var.get())
        except ValueError:
            num_samples = 50

        x, y = zip(*sorted(self.control_points))  # Sort by x values

        # Generate spline with regular sampling
        if len(x) >= 3:
            tck, u = splprep([x, y], s=0)
            t = np.linspace(0, 1, num_samples)
            x_new, y_new = splev(t, tck)
        else:
            # Just linear interpolation for 2 points
            t = np.linspace(0, 1, num_samples)
            x_new = np.interp(t, [0, 1], x)
            y_new = np.interp(t, [0, 1], y)

        # Create time series with sequential time values
        time_series = [(i, y_val) for i, y_val in enumerate(y_new)]

        # Display results
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "Time Series (t, x):\n")
        for t, val in time_series:
            self.result_text.insert(tk.END, f"({t}, {val:.2f})\n")

        return time_series


if __name__ == "__main__":
    root = tk.Tk()
    app = SplineDrawingApp(root)
    root.mainloop()
