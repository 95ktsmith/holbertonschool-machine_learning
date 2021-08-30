#!/usr/bin/env python3
""" Monte Carlo algorithm """
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1,
                gamma=0.99):
    """
    Performs the Monte Carlo algorithm
    env is the openAI environment instance
    V is a numpy.ndarray of shape (s,) containing the value estimate
    policy is a function that takes in a state and returns the next action
    episodes is the total number of episodes to train over
    max_steps is the maximum number of steps per episode
    alpha is the learning rate
    gamma is the discount rate
    Returns: V, the updated value estimate
    """
    for episode in range(episodes):
        state = env.reset()
        steps = []

        for step in range(max_steps):
            action = policy(state)
            new_state, reward, done, _ = env.step(action)
            steps.append([state, reward])

            if done is True:
                break

            state = new_state

        G = 0
        steps = np.array(steps, dtype=int)
        for step in steps[::-1]:

            state, reward = step
            G = gamma * G + reward

            if state not in steps[:episode, 0]:
                V[state] = V[state] + alpha * (G - V[state])

    return V
