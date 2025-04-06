import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class CSVPlotterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-View CSV Plotter")
        self.root.geometry("1000x800")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Control panel
        self.control_frame = ttk.Frame(root)
        self.control_frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(self.control_frame, text="Load CSV", command=self.load_csv).pack(
            side=tk.LEFT, padx=5, pady=5
        )

        # Plot selection
        self.plot_var = tk.IntVar(value=1)
        for i in range(1, 5):
            rb = ttk.Radiobutton(
                self.control_frame,
                text=f"Plot {i}",
                variable=self.plot_var,
                value=i,
                command=self.switch_plot,
            )
            rb.pack(side=tk.LEFT, padx=5, pady=5)

        self.file_label = ttk.Label(self.control_frame, text="No file loaded")
        self.file_label.pack(side=tk.LEFT, padx=5, pady=5)

        # Create plot layout frame
        self.plot_frame = ttk.Frame(root)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)

        # Initialize plots container
        self.plots = []
        self.canvases = []
        self.data_sources = [None] * 4
        self.current_view = 0

        # Create all four plot areas initially (but only show the first one)
        for i in range(4):
            fig, ax = plt.subplots(figsize=(8, 5))
            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas_widget = canvas.get_tk_widget()

            # Only show the first plot initially
            if i == 0:
                canvas_widget.pack(fill=tk.BOTH, expand=True)

            # Initialize plot
            ax.set_title(f"Plot View {i+1}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            ax.grid(True)

            # Show initial instructions
            ax.text(
                0.5,
                0.5,
                f"Click 'Load CSV' for Plot {i+1}",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
                fontsize=14,
            )
            canvas.draw()

            self.plots.append((fig, ax))
            self.canvases.append(canvas)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(
            root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def on_close(self):
        # Handle window close event properly
        plt.close("all")
        self.root.quit()
        self.root.destroy()
        sys.exit(0)

    def switch_plot(self):
        # Hide current plot
        self.canvases[self.current_view].get_tk_widget().pack_forget()

        # Show selected plot
        self.current_view = self.plot_var.get() - 1
        self.canvases[self.current_view].get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Update status bar to show information about the current plot
        data = self.data_sources[self.current_view]
        if data is not None:
            if self.is_heatmap_data(data):
                self.status_var.set(
                    f"Plot {self.current_view+1}: Heatmap visualization"
                )
            else:
                num_series = len(data.columns) - 1  # excluding time column
                self.status_var.set(
                    f"Plot {self.current_view+1}: {num_series} time series with {len(data)} points each"
                )
        else:
            self.status_var.set(f"Plot {self.current_view+1}: No data loaded")

    def load_csv(self):
        # Open file dialog to select CSV file
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Select CSV File for Plot " + str(self.current_view + 1),
        )

        if not file_path:
            return

        try:
            # Load the CSV file
            data = pd.read_csv(file_path)

            # Store data for the current plot
            self.data_sources[self.current_view] = data

            # Update file label
            filename = os.path.basename(file_path)
            self.file_label.config(text=f"Plot {self.current_view+1}: {filename}")

            # Check if the data is in heatmap format (row, column, value)
            if self.is_heatmap_data(data):
                self.plot_heatmap(data)
            else:
                # Check if the file has the expected format for time series
                if "t" not in data.columns:
                    messagebox.showwarning(
                        "Format Warning",
                        "CSV file doesn't contain a 't' column for time values. "
                        "First column will be used as time axis.",
                    )
                    # Use the first column as the time column
                    data = data.rename(columns={data.columns[0]: "t"})

                # Plot the time series data
                self.plot_time_series(data)

        except Exception as e:
            self.status_var.set(f"Error loading file: {str(e)}")
            messagebox.showerror("Error", f"Failed to load file: {str(e)}")

    def is_heatmap_data(self, data):
        """Check if data has row, column, value format suitable for heatmap"""
        required_cols = {"row", "column", "value"}
        return required_cols.issubset(set(map(str.lower, data.columns)))

    def plot_heatmap(self, data):
        """Plot data as a heatmap"""
        # Get current figure and axis
        fig, ax = self.plots[self.current_view]
        ax.clear()

        # Extract row, column, value columns (case-insensitive)
        cols = {col.lower(): col for col in data.columns}

        row_col = cols.get("row") or next(
            (c for c in data.columns if c.lower() == "row"), None
        )
        col_col = cols.get("column") or next(
            (c for c in data.columns if c.lower() == "column"), None
        )
        val_col = cols.get("value") or next(
            (c for c in data.columns if c.lower() == "value"), None
        )

        # Convert to matrix format
        pivot_data = data.pivot(index=row_col, columns=col_col, values=val_col)

        # Create heatmap
        im = ax.imshow(pivot_data, cmap="coolwarm")
        fig.colorbar(im, ax=ax, label=val_col)

        # Set plot properties
        ax.set_title(f"Heatmap Plot - Plot {self.current_view+1}")
        ax.set_xlabel(col_col)
        ax.set_ylabel(row_col)

        # If the matrix is not too large, add value labels
        if pivot_data.shape[0] <= 20 and pivot_data.shape[1] <= 20:
            for i in range(pivot_data.shape[0]):
                for j in range(pivot_data.shape[1]):
                    if not np.isnan(pivot_data.iloc[i, j]):
                        text = ax.text(
                            j,
                            i,
                            f"{pivot_data.iloc[i, j]:.2f}",
                            ha="center",
                            va="center",
                            color=(
                                "w"
                                if abs(pivot_data.iloc[i, j])
                                > (pivot_data.max().max() / 2)
                                else "black"
                            ),
                        )

        # Update the plot
        self.canvases[self.current_view].draw()

        # Update status
        self.status_var.set(
            f"Plot {self.current_view+1}: Heatmap visualization ({pivot_data.shape[0]}x{pivot_data.shape[1]})"
        )

    def plot_time_series(self, data):
        """Plot data as time series"""
        # Get current figure and axis
        fig, ax = self.plots[self.current_view]
        ax.clear()

        # Get time values
        time_values = data["t"]

        # Plot each data column except the time column
        for column in data.columns:
            if column != "t":
                ax.plot(time_values, data[column], label=column)

        # Set plot properties
        ax.set_title(f"Time Series Plot - Plot {self.current_view+1}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.grid(True)
        ax.legend()

        # Update the plot
        self.canvases[self.current_view].draw()

        # Update status
        num_series = len(data.columns) - 1  # excluding time column
        self.status_var.set(
            f"Plot {self.current_view+1}: {num_series} time series with {len(time_values)} points each"
        )


if __name__ == "__main__":
    root = tk.Tk()
    app = CSVPlotterApp(root)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        app.on_close()
