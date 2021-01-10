#!/usr/bin/env python3
""" Forward Propagation """
import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """ Creates forward propagation graph for the neural network
        x is the placeholder for the input data
        layer_sizes is a list containing the number of nodes in each layer
        activations is a list containing the activation functions for each
            layer of the network
        Returns the prediction of the network in tensor form
    """
    for i in range(0, len(layer_sizes)):
        layer = create_layer(x, layer_sizes[i], activations[i])
    return layer
