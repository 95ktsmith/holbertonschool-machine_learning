#!/usr/bin/env python3
""" Tensorflow layer with dropout """
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """ Creates a layer of a neural network using dropout
        prev is a tensor containing the output of the previous layer
        n is the number of nodes the new layer should contain
        activation is the activation function that should be used on the layer
        keep_prob is the probability that a node will be kept
        Returns: the output of the new layer
    """
    kinit = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(n, activation=activation, kernel_initializer=kinit)
    drop = tf.layers.Dropout(rate=keep_prob)
    return drop(layer(prev))
