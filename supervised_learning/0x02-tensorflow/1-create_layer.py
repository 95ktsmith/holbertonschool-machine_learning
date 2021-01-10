#!/usr/bin/env python3
""" Create Layer """
import tensorflow as tf


def create_layer(prev, n, activation):
    """ Create layer
        prev is the tensor ourput of the previous layer
        n is the number of nodes in the layer to create
        activation is the activation function that the layer should use
        Returns the tensor output of the layer
    """
    k_init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(n, name="layer", activation=activation,
                            kernel_initializer=k_init)
    return layer(prev)
