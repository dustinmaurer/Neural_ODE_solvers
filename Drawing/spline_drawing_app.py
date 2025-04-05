import csv
import sys
import tkinter as tk
from tkinter import filedialog, ttk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.interpolate import splev, splprep


class SplineDrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Line Spline Drawing App")
        self.root.geometry("800x600")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)  # Handle window close

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

        # Line management
        ttk.Button(self.control_frame, text="Add Line", command=self.add_new_line).pack(
            side=tk.LEFT, padx=5, pady=5
        )

        self.current_line_var = tk.StringVar(value="1")
        ttk.Label(self.control_frame, text="Current Line:").pack(
            side=tk.LEFT, padx=5, pady=5
        )
        self.line_dropdown = ttk.Combobox(
            self.control_frame, textvariable=self.current_line_var, width=5
        )
        self.line_dropdown.pack(side=tk.LEFT, padx=5, pady=5)
        self.line_dropdown["values"] = ("1",)
        self.line_dropdown.bind("<<ComboboxSelected>>", self.on_line_selected)

        ttk.Button(
            self.control_frame, text="Clear Current", command=self.clear_current_line
        ).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(
            self.control_frame, text="Clear All", command=self.clear_canvas
        ).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(
            self.control_frame, text="Generate CSV", command=self.save_to_csv
        ).pack(side=tk.LEFT, padx=5, pady=5)

        # Canvas for drawing
        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        # Initialize plot
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 100)
        self.ax.set_title("Click and drag to create control points")
        self.ax.grid(True)

        # Data storage - dictionary with line number as key
        self.all_lines = {
            1: {"control_points": [], "plot_handle": None, "spline_handle": None}
        }
        self.current_line = 1

        # Initialize first line
        (self.all_lines[1]["plot_handle"],) = self.ax.plot(
            [], [], "ro-", label=f"Line 1 control"
        )
        (self.all_lines[1]["spline_handle"],) = self.ax.plot(
            [], [], "-", label=f"Line 1 spline"
        )

        # Connect events
        self.canvas_mpl_connection = self.canvas.mpl_connect(
            "button_press_event", self.on_click
        )
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

        # Update legend
        self.ax.legend()
        self.canvas.draw()

    def on_close(self):
        # Handle window close event properly
        plt.close("all")  # Close all matplotlib figures
        self.root.quit()
        self.root.destroy()
        sys.exit(0)  # Force exit the Python process

    def on_click(self, event):
        if event.xdata is None or event.ydata is None:
            return

        # Check if clicking near an existing point in current line
        control_points = self.all_lines[self.current_line]["control_points"]
        for i, (x, y) in enumerate(control_points):
            if abs(x - event.xdata) < 3 and abs(y - event.ydata) < 3:
                self.dragging = True
                self.selected_point = i
                break
        else:
            # If not near any point, add a new one
            control_points.append((event.xdata, event.ydata))
            self.selected_point = len(control_points) - 1
            self.dragging = True

        self.update_plot()

    def on_move(self, event):
        if self.dragging and event.xdata is not None and event.ydata is not None:
            control_points = self.all_lines[self.current_line]["control_points"]
            control_points[self.selected_point] = (event.xdata, event.ydata)
            self.update_plot()

    def on_release(self, event):
        self.dragging = False
        self.selected_point = None

    def update_plot(self):
        control_points = self.all_lines[self.current_line]["control_points"]
        plot_handle = self.all_lines[self.current_line]["plot_handle"]
        spline_handle = self.all_lines[self.current_line]["spline_handle"]

        if len(control_points) > 0:
            x, y = zip(*control_points)
            plot_handle.set_data(x, y)

            # Update spline if we have enough points
            if len(control_points) >= 2:
                self.update_spline(self.current_line)

        self.canvas.draw_idle()  # Use draw_idle instead of draw for better performance

    def update_spline(self, line_num):
        control_points = self.all_lines[line_num]["control_points"]
        spline_handle = self.all_lines[line_num]["spline_handle"]

        if len(control_points) < 2:
            return

        x, y = zip(*sorted(control_points))  # Sort by x values

        # If we only have 2 points, just draw a line
        if len(x) == 2:
            spline_handle.set_data(x, y)
            return

        # Fit spline
        try:
            tck, u = splprep([x, y], s=0)
            u_new = np.linspace(0, 1, 100)
            x_new, y_new = splev(u_new, tck)

            spline_handle.set_data(x_new, y_new)
        except Exception as e:
            print(f"Error fitting spline: {e}")

    def clear_current_line(self):
        # Clear only the current line
        self.all_lines[self.current_line]["control_points"] = []
        self.all_lines[self.current_line]["plot_handle"].set_data([], [])
        self.all_lines[self.current_line]["spline_handle"].set_data([], [])
        self.result_text.delete(1.0, tk.END)
        self.canvas.draw()

    def clear_canvas(self):
        # Clear all lines
        for line_num in self.all_lines:
            self.all_lines[line_num]["control_points"] = []
            self.all_lines[line_num]["plot_handle"].set_data([], [])
            self.all_lines[line_num]["spline_handle"].set_data([], [])

        self.result_text.delete(1.0, tk.END)
        self.canvas.draw()

    def add_new_line(self):
        # Create a new line
        new_line_num = max(self.all_lines.keys()) + 1

        # Add new plot elements with different colors
        colors = ["r", "g", "b", "c", "m", "y", "k"]
        color_index = (new_line_num - 1) % len(colors)

        (plot_handle,) = self.ax.plot(
            [], [], f"{colors[color_index]}o-", label=f"Line {new_line_num} control"
        )
        (spline_handle,) = self.ax.plot(
            [], [], f"{colors[color_index]}-", label=f"Line {new_line_num} spline"
        )

        # Store in dictionary
        self.all_lines[new_line_num] = {
            "control_points": [],
            "plot_handle": plot_handle,
            "spline_handle": spline_handle,
        }

        # Update dropdown
        values = list(self.line_dropdown["values"])
        values.append(str(new_line_num))
        self.line_dropdown["values"] = tuple(values)

        # Switch to new line
        self.current_line = new_line_num
        self.current_line_var.set(str(new_line_num))

        # Update legend
        self.ax.legend()
        self.canvas.draw()

    def on_line_selected(self, event):
        try:
            self.current_line = int(self.current_line_var.get())
        except ValueError:
            pass

    def generate_time_series(self):
        try:
            num_samples = int(self.samples_var.get())
        except ValueError:
            num_samples = 50

        time_series_data = {}
        # Create time column
        time_series_data["t"] = list(range(num_samples))
        # max_time = num_samples - 1

        for line_num in self.all_lines:
            control_points = self.all_lines[line_num]["control_points"]

            if len(control_points) < 2:
                continue

            x, y = zip(*sorted(control_points))

            # Ensure first point is at x=0 and last point is at x=100
            # by adding boundary points if needed
            x_points = list(x)
            y_points = list(y)

            # Add point at x=0 if not present
            if min(x_points) > 0:
                # Extrapolate y value at x=0
                slope = (y_points[1] - y_points[0]) / (x_points[1] - x_points[0])
                y_at_zero = y_points[0] - (slope * x_points[0])
                x_points.insert(0, 0)
                y_points.insert(0, y_at_zero)

            # Add point at x=100 if not present
            if max(x_points) < 100:
                # Extrapolate y value at x=100
                slope = (y_points[-1] - y_points[-2]) / (x_points[-1] - x_points[-2])
                y_at_max = y_points[-1] + (slope * (100 - x_points[-1]))
                x_points.append(100)
                y_points.append(y_at_max)

            # Generate spline with regular sampling
            if len(x_points) >= 3:
                try:
                    # Normalize x points to 0-1 range for splprep
                    x_norm = [
                        (x - min(x_points)) / (max(x_points) - min(x_points))
                        for x in x_points
                    ]
                    tck, u = splprep([x_norm, y_points], s=0)
                    t = np.linspace(0, 1, num_samples)
                    x_new, y_new = splev(t, tck)

                    # Scale x values back
                    x_new = [
                        x * (max(x_points) - min(x_points)) + min(x_points)
                        for x in x_new
                    ]
                except Exception as e:
                    print(f"Error in spline calculation: {e}")
                    # Fall back to linear interpolation
                    t = np.linspace(0, 1, num_samples)
                    x_new = np.linspace(min(x_points), max(x_points), num_samples)
                    y_new = np.interp(x_new, x_points, y_points)
            else:
                # Just linear interpolation for 2 points
                t = np.linspace(0, 1, num_samples)
                x_new = np.linspace(min(x_points), max(x_points), num_samples)
                y_new = np.interp(x_new, x_points, y_points)

            # Create time series with sequential time values
            time_series = [(i, y_val) for i, y_val in zip(range(num_samples), y_new)]
            time_series_data[f"x{line_num}"] = [y for _, y in time_series]

        return time_series_data

    def save_to_csv(self):
        # Generate time series data
        time_series_data = self.generate_time_series()

        if (
            not time_series_data or len(time_series_data) <= 1
        ):  # Only "t" column means no lines
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(
                tk.END,
                "No valid lines to export. Add at least one line with 2+ points.",
            )
            return

        # Show preview in text box
        self.result_text.delete(1.0, tk.END)

        # Preview header
        header = ",".join(time_series_data.keys())
        self.result_text.insert(tk.END, f"{header}\n")

        # Preview first 5 rows
        for i in range(min(5, len(time_series_data["t"]))):
            row = [str(time_series_data[col][i]) for col in time_series_data.keys()]
            self.result_text.insert(tk.END, f"{','.join(row)}\n")

        self.result_text.insert(tk.END, "...\n")

        # Ask user for save location
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Save Time Series Data",
        )

        if not file_path:
            return

        # Write to CSV
        with open(file_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)

            # Write header
            writer.writerow(time_series_data.keys())

            # Write data rows
            for i in range(len(time_series_data["t"])):
                writer.writerow(
                    [time_series_data[col][i] for col in time_series_data.keys()]
                )

        self.result_text.insert(tk.END, f"\nData saved to {file_path}")


if __name__ == "__main__":
    root = tk.Tk()
    app = SplineDrawingApp(root)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        app.on_close()
