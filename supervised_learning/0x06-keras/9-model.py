#!/usr/bin/env python3
""" Save and Load Model """
import tensorflow.keras as K


def save_model(network, filename):
    """
    Saves a model
    network is the model to save
    filename is the path of the file that the model should be saved to
    """
    network.save(filename)


def load_model(filename):
    """
    Loads a model
    filename is the path of the file that the moodel should be loaded from
    """
    return K.models.load_model(filename)
