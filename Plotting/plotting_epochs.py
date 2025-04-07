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
        self.root.title("Trajectory and Weights Animator")
        self.root.geometry("1600x600")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Control panel
        self.control_frame = ttk.Frame(root)
        self.control_frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(
            self.control_frame, text="Load Trajectories", command=self.load_trajectories
        ).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(
            self.control_frame, text="Load Weights", command=self.load_weights
        ).pack(side=tk.LEFT, padx=5, pady=5)
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

        self.traj_label = ttk.Label(self.control_frame, text="No trajectories loaded")
        self.traj_label.pack(side=tk.LEFT, padx=5, pady=5)
        self.weights_label = ttk.Label(self.control_frame, text="No weights loaded")
        self.weights_label.pack(side=tk.LEFT, padx=5, pady=5)

        self.plot_frame = ttk.Frame(root)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)

        self.fig, (self.ax_traj, self.ax_weights) = plt.subplots(
            1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1, 1]}
        )
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.traj_data = None
        self.weights_data = None
        self.traj_ylim = None
        self.weights_vmin = None
        self.weights_vmax = None

        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(
            root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.plot_preview()

    def on_close(self):
        plt.close("all")
        self.root.quit()
        self.root.destroy()
        sys.exit(0)

    def load_trajectories(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Select Epoch Trajectories CSV",
        )
        if not file_path:
            return

        try:
            data = pd.read_csv(file_path)
            required_cols = {"epoch", "t", "trajectory", "value"}
            if not required_cols.issubset(data.columns):
                raise ValueError(
                    "Trajectories CSV must contain 'epoch', 't', 'trajectory', and 'value' columns"
                )
            self.traj_data = data
            self.traj_ylim = (
                data["value"].min(),
                data["value"].max(),
            )  # Set global y-axis limits
            self.traj_label.config(text=f"Traj: {os.path.basename(file_path)}")
            self.plot_preview()
            self.update_status()
        except Exception as e:
            self.status_var.set(f"Error loading trajectories: {str(e)}")
            messagebox.showerror("Error", f"Failed to load trajectories: {str(e)}")

    def load_weights(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Select Epoch Weights CSV",
        )
        if not file_path:
            return

        try:
            data = pd.read_csv(file_path)
            required_cols = {"epoch", "row", "column", "value"}
            if not required_cols.issubset(data.columns):
                raise ValueError(
                    "Weights CSV must contain 'epoch', 'row', 'column', and 'value' columns"
                )
            self.weights_data = data
            self.weights_vmin = data["value"].min()  # Set global color scale
            self.weights_vmax = data["value"].max()
            self.weights_label.config(text=f"Weights: {os.path.basename(file_path)}")
            self.plot_preview()
            self.update_status()
        except Exception as e:
            self.status_var.set(f"Error loading weights: {str(e)}")
            messagebox.showerror("Error", f"Failed to load weights: {str(e)}")

    def plot_preview(self):
        self.ax_traj.clear()
        self.ax_weights.clear()

        # Trajectory preview
        self.ax_traj.set_title("Trajectory Preview")
        self.ax_traj.set_xlabel("Time (t)")
        self.ax_traj.set_ylabel("Value")
        self.ax_traj.grid(True)
        if self.traj_data is not None:
            trajectories = self.traj_data["trajectory"].unique()
            for traj in trajectories:
                traj_data = self.traj_data[self.traj_data["trajectory"] == traj]
                self.ax_traj.plot(
                    traj_data["t"], traj_data["value"], label=f"Traj {traj}", alpha=0.5
                )
            self.ax_traj.set_ylim(self.traj_ylim)
            self.ax_traj.legend()
        else:
            self.ax_traj.text(
                0.5,
                0.5,
                "Load trajectories",
                ha="center",
                va="center",
                transform=self.ax_traj.transAxes,
            )

        # Weights preview
        self.ax_weights.set_title("Weights Preview")
        self.ax_weights.set_xlabel("Column")
        self.ax_weights.set_ylabel("Row")
        if self.weights_data is not None:
            final_epoch = self.weights_data["epoch"].max()
            weights = self.weights_data[self.weights_data["epoch"] == final_epoch]
            pivot = weights.pivot(index="row", columns="column", values="value")
            im = self.ax_weights.imshow(
                pivot, cmap="coolwarm", vmin=self.weights_vmin, vmax=self.weights_vmax
            )
            # self.fig.colorbar(im, ax=self.ax_weights)
            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    self.ax_weights.text(
                        j,
                        i,
                        f"{pivot.iloc[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="black",
                    )
        else:
            self.ax_weights.text(
                0.5,
                0.5,
                "Load weights",
                ha="center",
                va="center",
                transform=self.ax_weights.transAxes,
            )

        self.canvas.draw()

    def generate_gif(self):
        if self.traj_data is None or self.weights_data is None:
            messagebox.showwarning(
                "Warning", "Please load both trajectories and weights first"
            )
            return

        epochs = sorted(
            set(self.traj_data["epoch"].unique())
            & set(self.weights_data["epoch"].unique())
        )
        if not epochs:
            messagebox.showwarning(
                "Warning", "No common epochs between trajectories and weights"
            )
            return

        trajectories = self.traj_data["trajectory"].unique()

        def init():
            self.ax_traj.clear()
            self.ax_weights.clear()
            self.ax_traj.set_xlabel("Time (t)")
            self.ax_traj.set_ylabel("Value")
            self.ax_traj.set_ylim(self.traj_ylim)
            self.ax_traj.grid(True)
            self.ax_weights.set_xlabel("Column")
            self.ax_weights.set_ylabel("Row")
            return []

        def update(frame):
            epoch = epochs[frame]
            self.fig.suptitle(f"Epoch {epoch}", fontsize=16)

            # Update trajectory plot
            self.ax_traj.clear()
            self.ax_traj.set_xlabel("Time (t)")
            self.ax_traj.set_ylabel("Value")
            self.ax_traj.set_ylim(self.traj_ylim)
            self.ax_traj.grid(True)
            epoch_traj = self.traj_data[self.traj_data["epoch"] == epoch]
            self.ax_traj.set_xlim(self.traj_data["t"].min(), self.traj_data["t"].max())
            for traj in trajectories:
                traj_data = epoch_traj[epoch_traj["trajectory"] == traj]
                self.ax_traj.plot(
                    traj_data["t"], traj_data["value"], label=f"Traj {traj}"
                )
            self.ax_traj.legend()

            # Update weights heatmap
            self.ax_weights.clear()
            self.ax_weights.set_xlabel("Column")
            self.ax_weights.set_ylabel("Row")
            epoch_weights = self.weights_data[self.weights_data["epoch"] == epoch]
            pivot = epoch_weights.pivot(index="row", columns="column", values="value")
            im = self.ax_weights.imshow(
                pivot, cmap="coolwarm", vmin=self.weights_vmin, vmax=self.weights_vmax
            )
            # if frame == 0:
            #     self.fig.colorbar(im, ax=self.ax_weights)
            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    self.ax_weights.text(
                        j,
                        i,
                        f"{pivot.iloc[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="black",
                    )

            return []

        anim = FuncAnimation(
            self.fig,
            update,
            init_func=init,
            frames=len(epochs),
            interval=self.delay_var.get(),
            blit=False,
        )

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

    def update_status(self):
        status = []
        if self.traj_data is not None:
            epochs = len(self.traj_data["epoch"].unique())
            trajectories = len(self.traj_data["trajectory"].unique())
            status.append(f"Traj: {epochs} epochs, {trajectories} trajectories")
        if self.weights_data is not None:
            epochs = len(self.weights_data["epoch"].unique())
            rows = self.weights_data["row"].max() + 1
            cols = self.weights_data["column"].max() + 1
            status.append(f"Weights: {epochs} epochs, {rows}x{cols} matrix")
        self.status_var.set(" | ".join(status) if status else "Ready")


if __name__ == "__main__":
    root = tk.Tk()
    app = CSVPlotterApp(root)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        app.on_close()
