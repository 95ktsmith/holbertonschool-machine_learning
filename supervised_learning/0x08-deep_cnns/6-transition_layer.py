#!/usr/bin/env python3
""" Transition layer """
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    Builds a transition layer
    X is the output from the previous layer
    nb_filters is an integer representing the number of filters in X
    compression is the compression factor for the transition layer
    All weights use he normal intialization
    Returns the output of the transition layer and the number of filters
        within the output, respectively
    """
    norm = K.layers.BatchNormalization()(X)
    relu = K.layers.Activation('relu')(norm)
    conv = K.layers.Conv2D(
        filters=int(nb_filters * compression),
        kernel_size=1,
        strides=1,
        padding='same',
        kernel_initializer=K.initializers.he_normal()
    )(relu)
    pool = K.layers.AveragePooling2D()(conv)
    return pool, int(nb_filters * compression)
