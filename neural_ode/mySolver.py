"""
    This module contains functions relevant to solving ODEs
"""

import math
import numpy as np
from scipy.integrate import odeint
from scipy.special import expit

def get_trajectory(training_run, initial_conds, t=None):
  """ Solves the model in training_run for every weight matrix in the 'weight_trajectory' of training run

      training_run -- an object containing a 'model', and a 'weight_trajectory'
        model -- and ODE
        weight_trajectory -- a series of weight matrices that represent progress along the way from the initial weights to final weights
      inital_conds -- the initial values of each of the variables in the model
      t -- the time points that will be solved for by the ode solver (default np.linspace(0,30,50))
  """
  if t is None:
    t = np.linspace(0,30,50)
  # The solution of the trajectory of weights
  solution_trajectory = []
  for weights in training_run['weight_trajectory']:
    timecourse = odeint(training_run['model'], initial_conds, t, args = (weights,))
    solution_trajectory.append(timecourse)
  return solution_trajectory

def model(state, t, weights):
  """ This is the form of the ODE model 
      dY/dt = 1 / (1 + exp(- W.Y)) - Y
  
      state -- a list of initial conditions of length <n>
      weights -- a matrix of interaction weights of dimensions <n,n>
  """ 
  state = np.array(state).reshape(1,-1)
  new_state = expit(state.dot(weights)) - state
  new_state = new_state.tolist() # reshape the states coumn vector to a list <n>. Perhaps there is a better way of doing this without two 'reshape' operations.
  return new_state


