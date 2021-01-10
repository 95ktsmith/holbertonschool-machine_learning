#!/usr/bin/env python3
""" Placeholders """
import tensorflow as tf


def create_placeholders(nx, classes):
    """ Creates and returns two placeholders, x and y
        nx is the number of feature columns in our data
        classes is the number of classes in our classifier
            x is the placeholder for input data
            y is the placeholder for the one-hot labels for input data
    """
    x = tf.placeholder("float", [None, nx], "x")
    y = tf.placeholder("float", [None, classes], "y")
    return x, y
