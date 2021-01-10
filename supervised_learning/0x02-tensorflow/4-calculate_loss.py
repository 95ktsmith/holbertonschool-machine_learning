#!/usr/bin/env python3
""" Calculate Loss """
import tensorflow as tf


def calculate_loss(y, y_pred):
    """ Calculates the softmax cross entropy loss of a prediction
        y is a placeholder for the labels of the input data
        y_pred is a tensor containing the network's predictions
        Returns a tensor containing the loss of the prediction
    """
    return tf.losses.softmax_cross_entropy(y, y_pred)
