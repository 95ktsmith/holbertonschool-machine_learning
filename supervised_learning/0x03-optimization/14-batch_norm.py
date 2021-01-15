#!/usr/bin/env python3
""" Batch normalization """
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """ Creates a batch normalization layer for a neural network
        prev is the activated output of the previous layer
        n is the number of nodes in the layer to be created
        activation is the activation function that should be used on the
            output of the layer
        you should use the tf.layers.Dense layer as the base layer with kernal
            initializer
            tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
        your layer should incorporate two trainable parameters, gamma and beta,
            initialized as vectors of 1 and 0 respectively
        you should use an epsilon of 1e-8
        Returns: a tensor of the activated output for the layer
    """
    k_init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(n, name="layer", kernel_initializer=k_init)

    x = layer(prev)
    mean, std = tf.nn.moments(x, 0)
    gamma = tf.Variable(initial_value=tf.constant(1.0, shape=[n]),
                        trainable=True, name="gamma")
    beta = tf.Variable(initial_value=tf.constant(0.0, shape=[n]),
                       trainable=True, name="beta")
    norm_layer = tf.nn.batch_normalization(x, mean, std, beta, gamma, 1e-8)
    return activation(norm_layer)
