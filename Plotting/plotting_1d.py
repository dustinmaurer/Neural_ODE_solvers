import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class CSVPlotterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-View CSV Plotter with GIF Export")
        self.root.geometry("1000x800")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Control panel
        self.control_frame = ttk.Frame(root)
        self.control_frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(self.control_frame, text="Load CSV", command=self.load_csv).pack(
            side=tk.LEFT, padx=5, pady=5
        )
        ttk.Button(
            self.control_frame, text="Generate GIF", command=self.generate_gif
        ).pack(side=tk.LEFT, padx=5, pady=5)

        ttk.Label(self.control_frame, text="Frame Delay (ms):").pack(
            side=tk.LEFT, padx=5
        )
        self.delay_var = tk.IntVar(value=200)
        ttk.Entry(self.control_frame, textvariable=self.delay_var, width=5).pack(
            side=tk.LEFT, padx=5
        )

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

        self.plot_frame = ttk.Frame(root)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)

        self.plots = []
        self.canvases = []
        self.data_sources = [None] * 4
        self.current_view = 0

        for i in range(4):
            fig, ax = plt.subplots(figsize=(8, 5))
            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas_widget = canvas.get_tk_widget()
            if i == 0:
                canvas_widget.pack(fill=tk.BOTH, expand=True)

            ax.set_title(f"Plot View {i+1}")
            ax.set_xlabel("Time (t)")
            ax.set_ylabel("Value")
            ax.grid(True)
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

        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(
            root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def on_close(self):
        plt.close("all")
        self.root.quit()
        self.root.destroy()
        sys.exit(0)

    def switch_plot(self):
        self.canvases[self.current_view].get_tk_widget().pack_forget()
        self.current_view = self.plot_var.get() - 1
        self.canvases[self.current_view].get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.update_status()

    def load_csv(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Select CSV File for Plot " + str(self.current_view + 1),
        )
        if not file_path:
            return

        try:
            data = pd.read_csv(file_path)
            required_cols = {"epoch", "t", "trajectory", "value"}
            if not required_cols.issubset(data.columns):
                raise ValueError(
                    "CSV must contain 'epoch', 't', 'trajectory', and 'value' columns"
                )

            self.data_sources[self.current_view] = data
            filename = os.path.basename(file_path)
            self.file_label.config(text=f"Plot {self.current_view+1}: {filename}")
            self.plot_static_preview(data)
            self.update_status()

        except Exception as e:
            self.status_var.set(f"Error loading file: {str(e)}")
            messagebox.showerror("Error", f"Failed to load file: {str(e)}")

    def plot_static_preview(self, data):
        """Show a static preview of all epochs"""
        fig, ax = self.plots[self.current_view]
        ax.clear()

        trajectories = data["trajectory"].unique()
        for traj in trajectories:
            traj_data = data[data["trajectory"] == traj]
            ax.plot(traj_data["t"], traj_data["value"], label=f"Traj {traj}", alpha=0.5)

        ax.set_title(f"All Epochs Preview - Plot {self.current_view+1}")
        ax.set_xlabel("Time (t)")
        ax.set_ylabel("Value")
        ax.grid(True)
        ax.legend()
        self.canvases[self.current_view].draw()

    def generate_gif(self):
        """Generate and save a GIF animation"""
        data = self.data_sources[self.current_view]
        if data is None:
            messagebox.showwarning("Warning", "Please load a CSV file first")
            return

        epochs = sorted(data["epoch"].unique())
        trajectories = data["trajectory"].unique()

        fig, ax = plt.subplots(figsize=(10, 6))
        lines = {
            traj: ax.plot([], [], label=f"Traj {traj}")[0] for traj in trajectories
        }

        def init():
            ax.set_xlabel("Time (t)")
            ax.set_ylabel("Value")
            ax.set_title(f"Epoch {epochs[0]}")
            ax.grid(True)
            ax.legend()
            for line in lines.values():
                line.set_data([], [])
            return lines.values()

        def update(frame):
            epoch = epochs[frame]
            ax.set_title(f"Epoch {epoch}")
            epoch_data = data[data["epoch"] == epoch]

            # Set axis limits based on full data range
            if frame == 0:
                ax.set_xlim(data["t"].min(), data["t"].max())
                ax.set_ylim(data["value"].min(), data["value"].max())

            for traj in trajectories:
                traj_data = epoch_data[epoch_data["trajectory"] == traj]
                lines[traj].set_data(traj_data["t"], traj_data["value"])
            return lines.values()

        anim = FuncAnimation(
            fig,
            update,
            init_func=init,
            frames=len(epochs),
            interval=self.delay_var.get(),
            blit=True,
        )

        # Save the GIF
        save_path = filedialog.asksaveasfilename(
            defaultextension=".gif",
            filetypes=[("GIF files", "*.gif"), ("All files", "*.*")],
            title="Save GIF As",
        )
        if save_path:
            try:
                writer = PillowWriter(fps=1000 / self.delay_var.get())
                anim.save(save_path, writer=writer)
                messagebox.showinfo("Success", f"GIF saved to {save_path}")
                self.status_var.set(f"GIF saved: {os.path.basename(save_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save GIF: {str(e)}")
                self.status_var.set(f"Error saving GIF: {str(e)}")

        plt.close(fig)

    def update_status(self):
        data = self.data_sources[self.current_view]
        if data is not None:
            epochs = len(data["epoch"].unique())
            trajectories = len(data["trajectory"].unique())
            self.status_var.set(
                f"Plot {self.current_view+1}: {epochs} epochs, {trajectories} trajectories"
            )
        else:
            self.status_var.set(f"Plot {self.current_view+1}: No data loaded")


if __name__ == "__main__":
    root = tk.Tk()
    app = CSVPlotterApp(root)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        app.on_close()
