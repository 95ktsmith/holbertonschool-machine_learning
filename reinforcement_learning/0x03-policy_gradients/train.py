#!/usr/bin/env pythpn3
""" Train """

import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """
    Full training with policy gradients
    env: initial environment
    nb_episodes: number of episodes used for training
    alpha: the learning rate
    gamma: the discount factor
    Returns all values of the score
    """
    scores = []
    weight = np.random.rand(
        env.observation_space.shape[0],
        env.action_space.n
    )

    for episode in range(nb_episodes):
        state = env.reset()[None, :]

        grads = []
        rewards = []
        actions = []
        done = False

        while not done:
            if show_result is True and episode % 1000 == 0:
                env.render()

            # Get action and gradient from policy
            action, grad = policy_gradient(state, weight)

            # Update environment
            state, reward, done, _ = env.step(action)

            # Expand state dimensions
            state = state[None, :]

            # Append to episode history
            grads.append(grad)
            rewards.append(reward)
            actions.append(action)

        for i in range(len(grads)):
            # Calculate rewards from this step forward
            reward = sum([R * gamma ** R for R in rewards[i:]])

            # Apply gradients
            weight += alpha * grads[i] * reward

        scores.append(sum(rewards))

        print('Episode: {}     Score: {}'.format(episode, scores[episode]),
              end='\r', flush=False)

    return scores
