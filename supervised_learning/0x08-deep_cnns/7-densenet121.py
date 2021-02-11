#!/usr/bin/env python3
""" DenseNet-121 """
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    Builds a model of DenseNet-121 architecture
    growth_rate is the growth rate
    compression is the compression factor
    All weights use he normal intitialization
    Returns the keras model
    """
    input = K.layers.Input(shape=(224, 224, 3))

    norm_0 = K.layers.BatchNormalization()(input)
    relu_0 = K.layers.Activation('relu')(norm_0)
    conv_0 = K.layers.Conv2D(
        filters=growth_rate*2,
        kernel_size=7,
        strides=2,
        padding='same',
        kernel_initializer=K.initializers.he_normal()
    )(relu_0)
    pool_0 = K.layers.MaxPooling2D(
        pool_size=3,
        strides=2,
        padding='same'
    )(conv_0)

    block_0, n_filters = dense_block(pool_0, pool_0.shape[-1], growth_rate, 6)
    transition_0, n_filters = transition_layer(block_0, n_filters, compression)
    block_1, n_filters = dense_block(transition_0, n_filters, growth_rate, 12)
    transition_1, n_filters = transition_layer(block_1, n_filters, compression)
    block_2, n_filters = dense_block(transition_1, n_filters, growth_rate, 24)
    transition_2, n_filters = transition_layer(block_2, n_filters, compression)
    block_3, n_filters = dense_block(transition_2, n_filters, growth_rate, 16)

    pool_1 = K.layers.AveragePooling2D(
        pool_size=7
    )(block_3)

    output = K.layers.Dense(
        units=1000,
        activation='softmax',
        kernel_initializer=K.initializers.he_normal(),
    )(pool_1)

    return K.Model(inputs=input, outputs=output)
