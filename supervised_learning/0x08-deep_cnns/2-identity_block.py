#!/usr/bin/env python3
""" Identity block """
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """
    Builds an identity block
    A_prev is the output from the previous layer
    filters is a tuple or list containing F11, F3, F12, respectively:
        F11 is the number of filters in the first 1x1 convolution
        F3 is the number of filters in the 3x3 convolution
        F12 is the number of filters in the second 1x1 convolution
    All weights use he normal initialization
    Returns the activated output of the identity block
    """
    F11, F3, F12 = filters
    init = K.initializers.he_normal()

    conv_1a = K.layers.Conv2D(
        filters=F11,
        kernel_size=1,
        strides=1,
        padding='same',
        kernel_initializer=init
    )(A_prev)
    norm_1a = K.layers.BatchNormalization()(conv_1a)
    relu_1a = K.layers.Activation('relu')(norm_1a)

    conv_3 = K.layers.Conv2D(
        filters=F3,
        kernel_size=3,
        strides=1,
        padding='same',
        kernel_initializer=init
    )(relu_1a)
    norm_3 = K.layers.BatchNormalization()(conv_3)
    relu_3 = K.layers.Activation('relu')(norm_3)

    conv_1b = K.layers.Conv2D(
        filters=F12,
        kernel_size=1,
        strides=1,
        padding='same',
        kernel_initializer=init
    )(relu_3)
    norm_1b = K.layers.BatchNormalization()(conv_1b)

    added = K.layers.Add()([norm_1b, A_prev])
    return K.layers.Activation('relu')(added)
