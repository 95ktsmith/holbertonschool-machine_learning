#!/usr/bin/env python3
""" LeNet-5 in Tensorflow """
import tensorflow as tf


def lenet5(x, y):
    """
    Builds a modified version of the LeNet-5 architecture using tensorflow
    x is a tf.placeholder of shape (m, 28, 28, 1) containing the input images
        for the network
    m is the number of images
    y is a tf.placeholder of shape (m, 10) containing the one-hot labels for
        the network
    The model consists of the following layers in order:
        Convolutional layer with 6 kernels of shape 5x5 with same padding
        Max pooling layer with kernels of shape 2x2 with 2x2 strides
        Convolutional layer with 16 kernels of shape 5x5 with valid padding
        Max pooling layer with kernels of shape 2x2 with 2x2 strides
        Fully connected layer with 120 nodes
        Fully connected layer with 84 nodes
        Fully connected softmax output layer with 10 nodes
    All layers requiring initialization should initialize their kernels with
        the he_normal initialization method:
        tf.contrib.layers.variance_scaling_initializer()
    All hidden layers requiring activation should use relu activation
    Returns:
        A tensor for the softmax activated output
        A Training operation that utiliztes Adam optimization
        A tensor for the loss of the network
        A tensor for the accuracy of the network
    """
    k_init = tf.contrib.layers.variance_scaling_initializer()

    l1 = tf.layers.Conv2D(filters=6,
                          kernel_size=(5, 5),
                          padding='same',
                          kernel_initializer=k_init,
                          activation=tf.nn.relu)
    l2 = tf.layers.MaxPooling2D(pool_size=2,
                                strides=2)
    l3 = tf.layers.Conv2D(filters=16,
                          kernel_size=(5, 5),
                          padding='valid',
                          kernel_initializer=k_init,
                          activation=tf.nn.relu)
    l4 = tf.layers.MaxPooling2D(pool_size=2,
                                strides=2)
    l5 = tf.layers.Dense(units=120,
                         activation=tf.nn.relu,
                         kernel_initializer=k_init)
    l6 = tf.layers.Dense(units=84,
                         activation=tf.nn.relu,
                         kernel_initializer=k_init)
    l7 = tf.layers.Dense(units=10)

    l1_out = l1(x)
    l2_out = l2(l1_out)
    l3_out = l3(l2_out)
    l4_out = l4(l3_out)
    l5_out = l5(tf.layers.Flatten()(l4_out))
    l6_out = l6(l5_out)
    l7_out = l7(l6_out)
    y_pred = tf.nn.softmax(l7_out)

    prediction = tf.math.argmax(l7_out, axis=1)
    correct = tf.math.argmax(y, axis=1)
    equality = tf.math.equal(prediction, correct)
    accuracy = tf.math.reduce_mean(tf.cast(equality, tf.float32))

    loss = tf.losses.softmax_cross_entropy(y, l7_out)
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.apply_gradients(optimizer.compute_gradients(loss))

    return y_pred, training_op, loss, accuracy
