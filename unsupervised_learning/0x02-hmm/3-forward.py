#!/usr/bin/env python3
""" Forward Algorithm """
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    Performs forward algorithm for a hidden markov model
    """
    if type(Observation) is not np.ndarray or Observation.ndim != 1:
        return None, None
    if type(Emission) is not np.ndarray or Emission.ndim != 2:
        return None, None
    if type(Transition) is not np.ndarray or Transition.ndim != 2:
        return None, None
    if Transition.shape[0] != Transition.shape[1]:
        return None, None
    if Transition.shape[0] != Emission.shape[0]:
        return None, None
    if type(Initial) is not np.ndarray or Initial.ndim != 2:
        return None, None
    if Initial.shape[0] != Transition.shape[0] or Initial.shape[1] != 1:
        return None, None

    return None, None
