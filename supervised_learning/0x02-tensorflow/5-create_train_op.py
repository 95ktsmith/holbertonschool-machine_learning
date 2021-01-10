#!/usr/bin/env python3
""" Create train operation """
import tensorflow as tf


def create_train_op(loss, alpha):
    """ Creates the training operation for the network
        loss is the loss of the network's prediction
        alpha is the learning rate
        Returns an operation that trains the network using gradient descent
    """
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    return optimizer.apply_gradients(optimizer.compute_gradients(loss))
