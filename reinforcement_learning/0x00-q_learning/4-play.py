#!/usr/bin/env python3
""" Play an episode """
import numpy as np


def play(env, Q, max_steps=100):
    """
    Have an agent play an episode
    env is the FrozenLakeEnv instance
    Q is a numpy.ndarray containing the Q-table
    max_steps is the maximum number of steps in the episode
    Returns: The total reward for the episode
    """
    state = env.reset()

    for step in range(max_steps):
        env.render()
        action = np.argmax(Q[state])
        state, reward, done, info = env.step(action)

        if done is True:
            break

    env.render()
    return reward
