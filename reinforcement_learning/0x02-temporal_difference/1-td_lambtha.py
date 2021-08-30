#!/usr/bin/env python3
""" TD-Lambda Algorithm """
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """
    Performs the TD-lambda algorithm
    env is the openAI environment instance
    V is a numpy.ndarray of shape (s,) containing the value estimate
    lambtha is the eligibility trace factor
    policy is a function that takes in a state and returns the next action
    episodes is the total number of episodes to train over
    max_steps is the maximum number of steps per episode
    alpha is the learning rate
    gamma is the discount rate
    Returns: V, the updated value estimate
    """
    ET = np.zeros(V.shape[0])
    for episode in range(episodes):
        state = env.reset()

        for step in range(max_steps):
            action = policy(state)
            new_state, reward, done, _ = env.step(action)

            ET *= gamma * lambtha
            ET[state] += 1

            deltat = reward + gamma * V[new_state] - V[state]
            V += alpha * deltat * ET

            if done is True:
                break

            state = new_state

    return V
