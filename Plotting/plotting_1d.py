import os
import sys
import tkinter as tk
from tkinter import filedialog, ttk

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class CSVPlotterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Time Series CSV Plotter")
        self.root.geometry("800x600")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Control panel
        self.control_frame = ttk.Frame(root)
        self.control_frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(self.control_frame, text="Load CSV", command=self.load_csv).pack(
            side=tk.LEFT, padx=5, pady=5
        )
        self.file_label = ttk.Label(self.control_frame, text="No file loaded")
        self.file_label.pack(side=tk.LEFT, padx=5, pady=5)

        # Canvas for plotting
        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        # Initialize plot
        self.ax.set_title("Time Series Plot")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Value")
        self.ax.grid(True)

        # Data storage
        self.data = None

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(
            root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Show initial instructions
        self.ax.text(
            0.5,
            0.5,
            "Click 'Load CSV' to begin",
            horizontalalignment="center",
            verticalalignment="center",
            transform=self.ax.transAxes,
            fontsize=14,
        )
        self.canvas.draw()

    def on_close(self):
        # Handle window close event properly
        plt.close("all")
        self.root.quit()
        self.root.destroy()
        sys.exit(0)

    def load_csv(self):
        # Open file dialog to select CSV file
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Select Time Series CSV File",
        )

        if not file_path:
            return

        try:
            # Load the CSV file
            self.data = pd.read_csv(file_path)

            # Update file label
            filename = os.path.basename(file_path)
            self.file_label.config(text=filename)

            # Check if the file has the expected format
            if "t" not in self.data.columns:
                self.status_var.set(
                    "Error: CSV file must contain a 't' column for time values"
                )
                return

            # Plot the data
            self.plot_data()

        except Exception as e:
            self.status_var.set(f"Error loading file: {str(e)}")

    def plot_data(self):
        if self.data is None:
            return

        # Clear the plot
        self.ax.clear()

        # Get time values
        time_values = self.data["t"]

        # Plot each data column except the time column
        for column in self.data.columns:
            if column != "t":
                self.ax.plot(time_values, self.data[column], label=column)

        # Set plot properties
        self.ax.set_title("Time Series Plot")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Value")
        self.ax.grid(True)
        self.ax.legend()

        # Update the plot
        self.canvas.draw()

        # Update status
        num_series = len(self.data.columns) - 1  # excluding time column
        self.status_var.set(
            f"Loaded {num_series} time series with {len(time_values)} points each"
        )


if __name__ == "__main__":
    root = tk.Tk()
    app = CSVPlotterApp(root)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        app.on_close()
