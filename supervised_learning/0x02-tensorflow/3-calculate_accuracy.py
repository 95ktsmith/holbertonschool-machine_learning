#!/usr/bin/env python3
""" Calculate accuracy """
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """ Calculates the accuracy of a prediction
        y is a placeholder for the labels of the input data
        y_pred is a tensor containing the network's predicitions
        Returns a tensor containing the decmial accuracy of the prediction
    """
    return tf.math.reduce_mean(tf.cast(tf.math.equal(y_pred, y), tf.float32))
