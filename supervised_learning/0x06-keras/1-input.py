#!/usr/bin/env python3
""" Input model """
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ Builds a neural network
    nx is the number of input features to the network
    nx is the number of input features to the network
    layers is a list containing the number of nodes in each layer of the
        network
    activations is a list containing the activation functions used for each
        layer of the network
    lambtha is the L2 regularization parameter
    keep_prob is the probability that a node will be kept for dropout
    You are not allowed to use the Sequential class
    Returns: the keras model
    """
    inputs = K.Input(shape=(nx,))
    regularizer = K.regularizers.l2(lambtha)
    layer = K.layers.Dense(layers[0], activation=activations[0],
                           kernel_regularizer=regularizer)(inputs)
    for i in range(1, len(layers)):
        layer = K.layers.Dropout(1 - keep_prob)(layer)
        layer = K.layers.Dense(layers[i], activation=activations[i],
                               kernel_regularizer=regularizer)(layer)
    return K.Model(inputs=inputs, outputs=layer)
