#!/usr/bin/env python3
""" Momentum training operation """
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """ Creates training operation for a nerual network using gradient descent
            with momentum optimization
        loss is the loss of the network
        alpha is the learning rate
        beta1 is the momentum weight
        returns the momentum operation
    """
    optimizer = tf.train.MomentumOptimizer(alpha, beta1)
    return optimizer.apply_gradients(optimizer.compute_gradients(loss))
