#!/usr/bin/env python3
""" Inception Network """
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    Builds an inception network
    Returns the keras model
    """
    input = K.Input(shape=(224, 224, 3))

    conv_1 = K.layers.Conv2D(
        filters=64,
        kernel_size=7,
        strides=2,
        padding='same',
        activation='relu'
    )(input)

    pool_1 = K.layers.MaxPooling2D(
        pool_size=3,
        strides=2,
        padding='same'
    )(conv_1)

    conv_2 = K.layers.Conv2D(
        filters=64,
        kernel_size=1,
        strides=1,
        padding='valid',
        activation='relu'
    )(pool_1)

    conv_3 = K.layers.Conv2D(
        filters=192,
        kernel_size=3,
        strides=1,
        padding='same',
        activation='relu'
    )(conv_2)

    pool_2 = K.layers.MaxPooling2D(
        pool_size=3,
        strides=2,
        padding='same'
    )(conv_3)

    inception_1 = inception_block(pool_2, (64, 96, 128, 16, 32, 32))
    inception_2 = inception_block(inception_1, (128, 128, 192, 32, 96, 64))

    pool_3 = K.layers.MaxPooling2D(
        pool_size=3,
        strides=2,
        padding='same'
    )(inception_2)

    inception_3 = inception_block(pool_3, (192, 96, 208, 16, 48, 64))
    inception_4 = inception_block(inception_3, (160, 112, 224, 24, 64, 64))
    inception_5 = inception_block(inception_4, (128, 128, 256, 24, 64, 64))
    inception_6 = inception_block(inception_5, (112, 144, 288, 32, 64, 64))
    inception_7 = inception_block(inception_6, (256, 160, 320, 32, 128, 128))

    pool_4 = K.layers.MaxPooling2D(
        pool_size=3,
        strides=2,
        padding='same',
    )(inception_7)

    inception_8 = inception_block(pool_4, (256, 160, 320, 32, 128, 128))
    inception_9 = inception_block(inception_8, (384, 192, 384, 48, 128, 128))

    pool_5 = K.layers.AveragePooling2D(
        pool_size=7,
        strides=1,
        padding='valid'
    )(inception_9)

    dropout = K.layers.Dropout(rate=0.4)(pool_5)
    linear = K.layers.Dense(units=1000, activation='softmax')(dropout)

    return K.Model(inputs=input, outputs=linear)
