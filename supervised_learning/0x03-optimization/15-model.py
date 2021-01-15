#!/usr/bin/env python3
""" Model """
import tensorflow as tf
import numpy as np
create_batch_norm_layer = __import__('14-batch_norm').create_batch_norm_layer
create_Adam_op = __import__('10-Adam').create_Adam_op
learning_rate_decay = __import__('12-learning_rate_decay').learning_rate_decay
shuffle_data = __import__('2-shuffle_data').shuffle_data


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


def create_layer(prev, n, activation):
    """ Create layer
        prev is the tensor ourput of the previous layer
        n is the number of nodes in the layer to create
        activation is the activation function that the layer should use
        Returns the tensor output of the layer
    """
    k_init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(n, name="layer", activation=activation,
                            kernel_initializer=k_init)
    return layer(prev)


def forward_prop(x, layer_sizes=[], activations=[]):
    """ Creates forward propagation graph for the neural network
        x is the placeholder for the input data
        layer_sizes is a list containing the number of nodes in each layer
        activations is a list containing the activation functions for each
            layer of the network
        Returns the prediction of the network in tensor form
    """
    layer = create_batch_norm_layer(x, layer_sizes[0], activations[0])
    for i in range(1, len(layer_sizes) - 1):
        layer = create_batch_norm_layer(layer, layer_sizes[i], activations[i])
    layer = create_layer(layer, layer_sizes[-1], activations[-1])
    return layer


def calculate_loss(y, y_pred):
    """ Calculates the softmax cross entropy loss of a prediction
        y is a placeholder for the labels of the input data
        y_pred is a tensor containing the network's predictions
        Returns a tensor containing the loss of the prediction
    """
    return tf.losses.softmax_cross_entropy(y, y_pred)


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


def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9,
          beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    """ Builds, trains, and saves a neural network model in tensorflow using
            Adam optimization, mini-batch gradient descent, learning rate
            decay, and batch normalization
        Data_train is a tuple containing the training inputs and training
            labels, respectively
        Data_valid is a tuple containing the validation inputs and validation
            labels, respectively
        layers is a list containing the number of nodes in each layer of the
            network
        activation is a list containing the activation functions used for each
            layer of the network
        alpha is the learning rate
        beta1 is the weight for the first moment of Adam Optimization
        beta2 is the weight for the second moment of Adam Optimization
        epsilon is a small number used to avoid division by zero
        decay_rate is the decay rate for inverse time decay of the learning
            rate (the corresponding decay step should be 1)
        batch_size is the number of data points that should be in a mini-batch
        epochs is the number of times the training should pass through the
            whole dataset
        save_path is the path where the model should be saved to
        Returns: the path where the model was saved
    """
    x, y = create_placeholders(Data_train[0].shape[1], Data_train[1].shape[1])
    y_pred = forward_prop(x, layers, activations)
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    train_op = create_Adam_op(loss, alpha, beta1, beta2, epsilon)
    global_step = tf.Variable(0, trainable=False)
    alpha = learning_rate_decay(alpha, decay_rate, global_step, 1)

    tf.add_to_collection("x", x)
    tf.add_to_collection("y", y)
    tf.add_to_collection("y_pred", y_pred)
    tf.add_to_collection("loss", loss)
    tf.add_to_collection("accuracy", accuracy)
    tf.add_to_collection("train_op", train_op)
    tf.add_to_collection("global_step", global_step)
    tf.add_to_collection("alpha", alpha)

    train_dict = {x: Data_train[0], y: Data_train[1]}
    valid_dict = {x: Data_valid[0], y: Data_valid[1]}

    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):
            epoch_train_cost = sess.run(loss, feed_dict=train_dict)
            epoch_train_acc = sess.run(accuracy, feed_dict=train_dict)
            epoch_valid_cost = sess.run(loss, feed_dict=valid_dict)
            epoch_valid_acc = sess.run(accuracy, feed_dict=valid_dict)
            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(epoch_train_cost))
            print("\tTraining Accuracy: {}".format(epoch_train_acc))
            print("\tValidation Cost: {}".format(epoch_valid_cost))
            print("\tValidation Accuracy: {}".format(epoch_valid_acc))

            X_shuffled, Y_shuffled = shuffle_data(Data_train[0], Data_train[1])
            for batch in range(int(Data_train[0].shape[0] / batch_size + 1)):
                if batch > 0 and batch % 100 == 0:
                    step_cost = sess.run(loss, feed_dict=mini_dict)
                    step_acc = sess.run(accuracy, feed_dict=mini_dict)
                    print("\tStep {}:".format(batch))
                    print("\t\tCost: {}".format(step_cost))
                    print("\t\tAccuracy: {}".format(step_acc))
                start = batch * batch_size
                end = (batch + 1) * batch_size
                mini_dict = {x: X_shuffled[start:end],
                             y: Y_shuffled[start:end]}
                sess.run(train_op, feed_dict=mini_dict)
            sess.run(tf.assign(global_step, epoch + 1))

        epoch_train_cost = sess.run(loss, feed_dict=train_dict)
        epoch_train_acc = sess.run(accuracy, feed_dict=train_dict)
        epoch_valid_cost = sess.run(loss, feed_dict=valid_dict)
        epoch_valid_acc = sess.run(accuracy, feed_dict=valid_dict)
        print("After {} epochs:".format(epochs))
        print("\tTraining Cost: {}".format(epoch_train_cost))
        print("\tTraining Accuracy: {}".format(epoch_train_acc))
        print("\tValidation Cost: {}".format(epoch_valid_cost))
        print("\tValidation Accuracy: {}".format(epoch_valid_acc))
        saved = saver.save(sess, save_path)
    return saved
