#!/usr/bin/env python3
""" Inception block """
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
    Builds an inception block
    A_prev is the output from the previous layer
    filters is a tuple/list containing F1, F3R, F3,F5R, F5, FPP, respectively:
        F1 is the number of filters in the 1x1 convolution
        F3R is the number of filters in the 1x1 convolution before the 3x3
            convolution
        F3 is the number of filters in the 3x3 convolution
        F5R is the number of filters in the 1x1 convolution before the 5x5
            convolution
        F5 is the number of filters in the 5x5 convolution
        FPP is the number of filters in the 1x1 convolution after the max
            pooling
    Returns: the concatenated output of the inception block
    """
    F1, F3R, F3, F5R, F5, FPP = filters

    F1_C = K.layers.Conv2D(
        filters=F1,
        kernel_size=1,
        padding="same",
        activation="relu"
    )(A_prev)

    F3_R = K.layers.Conv2D(
        filters=F3R,
        kernel_size=1,
        padding="same",
        activation="relu"
    )(A_prev)

    F3_C = K.layers.Conv2D(
        filters=F3,
        kernel_size=3,
        padding="same",
        activation="relu"
    )(F3_R)

    F5_R = K.layers.Conv2D(
        filters=F5R,
        kernel_size=1,
        padding="same",
        activation="relu"
    )(A_prev)

    F5_C = K.layers.Conv2D(
        filters=F5,
        kernel_size=5,
        padding="same",
        activation="relu"
    )(F5_R)

    F3_P = K.layers.MaxPooling2D(
        pool_size=3,
        strides=1,
        padding="same"
    )(A_prev)

    FP_R = K.layers.Conv2D(
        filters=FPP,
        kernel_size=1,
        padding="same",
        activation="relu"
    )(F3_P)

    return K.layers.Concatenate()([F1_C, F3_C, F5_C, FP_R])
