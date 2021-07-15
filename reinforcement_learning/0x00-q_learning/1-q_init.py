#!/usr/bin/env python3
""" Initialize Q-table """
import numpy as np


def q_init(env):
    """
    Initializes a Q-table
    env: the FrozenLakeEnv instance
    Returns: The Q-table as a numpy.ndaray of zeros
    """
    action_space_Size = env.action_space.n
    state_space_size = env.observation_space.n
    q_table = np.zeros((state_space_size, action_space_Size))
    return q_table
