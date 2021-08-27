#!/usr/bin/env python3
""" Trains an agent to play Breakout """
import numpy as np
import gym

import keras as K
from keras.layers import Input, Conv2D, Flatten, Dense
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy

env = gym.make('Breakout-v0')
env.render()
nb_actions = env.action_space.n

inputs = Input(shape=((4,) + env.observation_space.shape))
x = Conv2D(32, 8, strides=4, activation='relu')(inputs)
x = Conv2D(64, 4, strides=2, activation='relu')(x)
x = Conv2D(64, 2, strides=1, activation='relu')(x)
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
action = Dense(nb_actions, activation='linear')(x)
model = K.Model(inputs=inputs, outputs=action)
model.summary()

memory = SequentialMemory(limit=500000, window_length=3)
policy = EpsGreedyQPolicy()
dqn = DQNAgent(
    model=model,
    nb_actions=nb_actions,
    memory=memory,
    nb_steps_warmup=10,
    target_model_update=1e-2,
    policy=policy
)
dqn.compile(K.optimizers.Adam(lr=1e-3), metrics=['mae'])

dqn.fit(env, nb_steps=50000, log_interval=1000, visualize=False, verbose=2)
dqn.save_weights('policy.h5', overwrite=True)
