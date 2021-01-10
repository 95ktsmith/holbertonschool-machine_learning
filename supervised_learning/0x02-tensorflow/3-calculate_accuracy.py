#!/usr/bin/env python3
""" Calculate accuracy """
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """ Calculates the accuracy of a prediction
        y is a placeholder for the labels of the input data
        y_pred is a tensor containing the network's predicitions
        Returns a tensor containing the decmial accuracy of the prediction
    """
    prediction = tf.math.argmax(y_pred, axis=1)
    correct = tf.math.argmax(y, axis=1)
    equality = tf.math.equal(prediction, correct)
    return tf.math.reduce_mean(tf.cast(equality, tf.float32))
