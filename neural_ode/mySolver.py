"""
    This module contains functions relevant to solving ODEs
"""

import numpy as np
from scipy.integrate import odeint
from scipy.special import expit
import matplotlib.pyplot as plt

class mySolution:
  def __init__(self, data, t, nCells=1, nGenes=None):
    self.data = data
    self.t = t
    self.nCells = nCells
    
    if nGenes == None:
      self.nGenes = data.shape[1]
    else:
      self.nGenes = nGenes
  
    self.est_weights = None
    self.weight_trajectory = []

  def model(self, states, t, weights):
    states_array = np.array(states).reshape(-1,1)
    new_states = expit( weights.dot(states_array) ) - states_array
    out = new_states.reshape(-1).tolist()
    return out

  def solve_weights(self, est_weights=None, learning_rate=1.0, epochs=5, report_rate=5, verbose=False):
    t_bounds = np.append(self.t[:-1].reshape(-1,1), self.t[1:].reshape(-1,1), 1)
    t_dif = (t_bounds[:,1] - t_bounds[:,0]).reshape(-1,1)
    inputs = self.data[:-1]
    outputs = self.data[1:]

    # Initialize weights
    if est_weights == None:
      if self.est_weights == None:
        self.est_weights = np.random.normal(loc = 0, scale = 1 / self.nGenes, size = (self.nGenes,self.nGenes) )
    else:
      self.est_weights = est_weights

    self.weight_trajectory.append(self.est_weights)

    for epoch in range(epochs):
      predictions = outputs * 0
      for i in range(len(inputs)):
        predictions[i] = odeint(self.model, inputs[i], t_bounds[i], args = (self.est_weights,))[1] # solve the model for the outputs given the est_weights
        
      errors = (predictions - outputs) / t_dif
      out_grad = - (predictions * (1 - predictions) * errors)
      delta_w = out_grad.T.dot(inputs)

      self.est_weights = self.est_weights + delta_w * learning_rate

      # Record the trajectory of the weights and report the error
      if(epoch % report_rate == (report_rate -1)  or epoch == (epochs-1)): 
        self.weight_trajectory.append(self.est_weights)
        if verbose:
          print("Epoch {}, abs error of : {}".format(epoch+1, np.abs(errors).mean()))

  def get_trajectory(self):
    """ Solves the model in training_run for every weight matrix in the 'weight_trajectory' of training run

        training_run -- an object containing a 'model', and a 'weight_trajectory'
          model -- and ODE
          weight_trajectory -- a series of weight matrices that represent progress along the way from the initial weights to final weights
        inital_conds -- the initial values of each of the variables in the model
    """
    # The solution of the trajectory of weights
    solution_trajectory = []
    for weights in self.weight_trajectory:
      timecourse = odeint(self.model, self.data[0], self.t, args = (weights,))
      solution_trajectory.append(timecourse)
    return solution_trajectory

  def plot_ode(self, figsize=(12, 6), vline=False):
    """ Plot a simple line graph for a matrix of time-series. """
    fig, axes = plt.subplots(ncols = self.nCells, nrows = 1, figsize=figsize, constrained_layout = True)
    if (self.nCells == 1):
      axes.plot(self.t, self.data)
      axes.set_title('Network Trajectory')
      axes.set_xlabel('time')
      axes.set_ylabel('y(t)')
      axes.legend([i+1 for i in range(self.nGenes)], loc = 'upper left')
      axes.set_ylim(0,1)
      if vline: 
        for time in t: 
          axes.axvline(time, color = 'black', alpha = 0.5, ls = '--')
    else:  
      for i,ax in enumerate(axes):
        ax.plot(self.t, self.data[:,(i*self.nGenes):((1+i)*self.nGenes)])
        ax.set_title('Network Trajectory')
        ax.set_xlabel('time')
        ax.set_ylabel('y(t)')
        ax.legend([i+1 for i in range(self.nGenes)], loc = 'upper left')
        ax.set_ylim(0,1)
    fig.show()
