###
###
### This module contains functions relevant to:
### Animating solutions
###
###
import matplotlib.pyplot as plt
from matplotlib import animation, rc

def animate_heatmap(weight_trajectory, weight_bounds=3):
    """ Calls the solution_trajectory function, then creates an animated line plot of the solutions

        training_run -- an object containing a 'model', and a 'weight_trajectory'
        model -- and ODE
        weight_trajectory -- a series of weight matrices that represent progress along the way from the initial weights to final weights
        inital_conds -- the initial values of each of the variables in the model
    """
    fig, ax = plt.subplots(figsize = (6,6))
    fig.subplots_adjust(left=0, bottom=0.1, right=1, top=.9, wspace=None, hspace=None)
    # ax.set_title('Weight Trajectory')
    # ax.set_xlabel('Connections In')
    # ax.set_ylabel('Connections Out')
    ax = plt.imshow(weight_trajectory[0], cmap="coolwarm", vmin = -weight_bounds, vmax = weight_bounds)
   
    class ani:
        n=0
    def update(frame):
        ax.set_array(weight_trajectory[ani.n-2])
        ani.n = ani.n+1
        return ax,

    ani = animation.FuncAnimation(fig, update, interval=200, blit=True, frames= len(weight_trajectory)-1 )
    plt.close()
    return ani