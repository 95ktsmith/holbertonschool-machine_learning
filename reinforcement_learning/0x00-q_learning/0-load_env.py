#!/usr/bin/env python3
""" Load Environment """
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Loads a premade FrozenLakeEnv environment from OpenAI's gym
    desc: None, or a list of lists containing a custom description of the map
        to load for the enironment.
    map_name: None or a string containing the pre-made map to load
    is_slippery: boolean to determine if the ice is slippery
    Returns: The environment
    """
    env = FrozenLakeEnv(desc, map_name, is_slippery)
    return env
