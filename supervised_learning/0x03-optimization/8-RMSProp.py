#!/usr/bin/env python3
""" RMSProp training op """
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """ Creates the training operation for a neural network using RMSProp
            optimization
        loss is the loss of the network
        alpha is the learning rate
        beta2 is the RMSProp weight
        epsilon is a small number to avoid division by zero
        Returns: the RMSProp optimization operation
    """
    optimizer = tf.train.RMSPropOptimizer(alpha, momentum=beta2,
                                          epsilon=epsilon)
    return optimizer.apply_gradients(optimizer.compute_gradients(loss))
