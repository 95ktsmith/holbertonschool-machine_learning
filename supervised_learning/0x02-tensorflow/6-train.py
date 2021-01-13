#!/usr/bin/env python3
""" Train """
import tensorflow as tf
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations,
          alpha, iterations, save_path="/tmp/model.ckpt"):
    """ Builds, trains, and saves a neural network classifier
        X-train is a numpy.ndarray containing the train input data
        Y_train is a numpy.ndarray containing the training labels
        X_valid is a numpy.ndarray containing the validation input data
        Y_valid is a numpy.ndarray containing the validation labels
        layer_sizes is a list containing the number of nodes of each layer
            in the network
        activations is a list containing the activation functions for each
            layer of the network
        alpha is the learning rate
        iterations is the number of iterations to train over
        save_path designates where to save the model
    """
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    y_pred = forward_prop(x, layer_sizes, activations)
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    train_op = create_train_op(loss, alpha)

    tf.add_to_collection("x", x)
    tf.add_to_collection("y", y)
    tf.add_to_collection("y_pred", y_pred)
    tf.add_to_collection("loss", loss)
    tf.add_to_collection("accuracy", accuracy)
    tf.add_to_collection("train_op", train_op)

    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(iterations):
        if i == 0 or i % 100 == 0:
            train_pred = sess.run(y_pred, feed_dict={x: X_train})
            train_loss = sess.run(loss, feed_dict={y_pred: train_pred,
                                                   y: Y_train})
            train_acc = sess.run(accuracy, feed_dict={y: Y_train,
                                                      y_pred: train_pred})
            valid_pred = sess.run(y_pred, feed_dict={x: X_valid})
            valid_loss = sess.run(loss, feed_dict={y_pred: valid_pred,
                                                   y: Y_valid})
            valid_acc = sess.run(accuracy, feed_dict={y: Y_valid,
                                                      y_pred: valid_pred})

            print("After {} iterations:".format(i))
            print("\tTraining Cost: {}".format(train_loss))
            print("\tTraining Accuracy: {}".format(train_acc))
            print("\tValidation Cost: {}".format(valid_loss))
            print("\tValidation Accuracy: {}".format(valid_acc))

        sess.run(train_op, feed_dict={x: X_train, y: Y_train})

    train_pred = sess.run(y_pred, feed_dict={x: X_train})
    train_loss = sess.run(loss, feed_dict={y_pred: train_pred,
                                           y: Y_train})
    training_acc = sess.run(accuracy, feed_dict={y: Y_train,
                                                 y_pred: train_pred})
    valid_pred = sess.run(y_pred, feed_dict={x: X_valid})
    valid_loss = sess.run(loss, feed_dict={y_pred: valid_pred,
                                           y: Y_valid})
    valid_acc = sess.run(accuracy, feed_dict={y: Y_valid,
                                              y_pred: valid_pred})

    print("After {} iterations:".format(iterations))
    print("\tTraining Cost: {}".format(training_loss))
    print("\tTraining Accuracy: {}".format(training_acc))
    print("\tValidation Cost: {}".format(valid_loss))
    print("\tValidation Accuracy: {}".format(valid_acc))

    saver.save(sess, save_path)
    sess.close()
    return save_path
