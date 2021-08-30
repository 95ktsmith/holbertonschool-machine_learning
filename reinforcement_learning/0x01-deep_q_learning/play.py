#!/usr/bin/env python3
""" Loads weights for an agent to play Breakout """
import numpy as np
import gym

import tensorflow.keras as K
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Permute
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.callbacks import FileLogger
from rl.core import Processor
from PIL import Image


INPUT_SHAPE = (84, 84)


class AtariProcessor(Processor):
    """ Preprocess images as per deepmind paper """
    def process_observation(self, observation):
        """ Process observation """
        assert observation.ndim == 3
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE).convert('L')
        processed_observation = np.array(img)
        assert processed_observation.shape == INPUT_SHAPE
        return processed_observation.astype('uint8')

    def process_state_batch(self, batch):
        """ Convert batch to float32 """
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        """ Process reward """
        return np.clip(reward, -1., 1.)


env = gym.make('Breakout-v0')
env.reset()
nb_actions = env.action_space.n

inputs = Input(shape=((4,) + INPUT_SHAPE))
x = Permute((2, 3, 1))(inputs)
x = Conv2D(32, 8, strides=4, activation='relu')(x)
x = Conv2D(64, 4, strides=2, activation='relu')(x)
x = Conv2D(64, 2, strides=3, activation='relu')(x)
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
action = Dense(nb_actions, activation='linear')(x)
model = K.Model(inputs=inputs, outputs=action)
model.summary()

memory = SequentialMemory(limit=1000000, window_length=4)
policy = LinearAnnealedPolicy(
    EpsGreedyQPolicy(),
    attr='eps',
    value_max=1.,
    value_min=.1,
    value_test=0.05,
    nb_steps=1000000
)
dqn = DQNAgent(
    model=model,
    nb_actions=nb_actions,
    memory=memory,
    nb_steps_warmup=50000,
    target_model_update=10000,
    policy=policy,
    processor=AtariProcessor(),
    gamma=.99,
    train_interval=4,
    delta_clip=1.
)
dqn.compile(K.optimizers.Adam(lr=.00025), metrics=['mae'])

dqn.load_weights('policy.h5')

dqn.test(env, nb_episodes=10, visualize=True)
