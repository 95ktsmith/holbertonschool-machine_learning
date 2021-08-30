#!/usr/bin/env python3
""" SARSA Lambda algorithm """

import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs SARSA(lambda) algorithm
    env is the openAI environment instance
    Q is a numpy.ndarray of shape (s,a) containing the Q table
    lambtha is the eligibility trace factor
    episodes is the total number of episodes to train over
    max_steps is the maximum number of steps per episode
    alpha is the learning rate
    gamma is the discount rate
    epsilon is the initial threshold for epsilon greedy
    min_epsilon is the minimum value that epsilon should decay to
    epsilon_decay is the decay rate for updating epsilon between episodes
    Returns: Q, the updated Q table
    """
    ET = np.zeros(Q.shape)

    for episode in range(episodes):
        state = env.reset()
        action = policy(state, Q, epsilon)

        for step in range(max_steps):
            new_state, reward, done, _ = env.step(action)
            new_action = policy(new_state, Q, epsilon)

            ET *= gamma * lambtha
            ET[state, action] += 1

            deltat = reward + gamma * Q[new_state, new_action] -\
                Q[state, action]

            Q += alpha * deltat * ET

            if done is True:
                break

            state = new_state
            action = new_action

        epsilon = (min_epsilon +
                   (epsilon - min_epsilon) * np.exp(-epsilon_decay * episode))

    return Q


def policy(state, Q, epsilon):
    """
    Epsilon greedy policy
    """
    p = np.random.uniform()

    if p > epsilon:
        action = np.argmax(Q[state])
    else:
        action = np.random.randint(Q.shape[1])

    return action
