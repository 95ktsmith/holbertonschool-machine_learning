#!/usr/bin/env python3
""" Predict """
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    Makes a prediction using a neural network
    network is the network model to make the prediction with
    data is the unput data to make the prediction with
    verbose is a boolean that determines if output should be printed during the
        prediction process
    """
    return network.predict(data, verbose=verbose)
