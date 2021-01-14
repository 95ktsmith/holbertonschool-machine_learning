#!/usr/bin/env python3
""" Mini batch training """
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """ Trains a loaded netork model using mini-batch gradient descent
        X_train is a numpy.ndarray of shape (m, 784) containing the training
            data
        m is the number of data points
        784 is the number of input features
        Y_train is a one-hot numpy.ndarray of shape (m, 10) containing the
            training labels
        10 is the number of classes the model should classify
        X_valid is a numpy.ndarray of shape (m, 784) containing the validation
            data
        Y_valid is a one-hot numpy.ndarray of shape (m, 10) containing the
            validation labels
        batch_size is the number of data points in a batch
        epochs is the number of times the training should pass through the
            whole dataset
        load_path is the path from which to load the model
        save_path is the path to where the model should be saved after training
        Returns: the path where the model was saved
    """
    with tf.Session() as sess:
        loader = tf.train.import_meta_graph(load_path + ".meta")
        loader.restore(sess, load_path)

        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]
        train_op = tf.get_collection("train_op")[0]

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        train_dict = {x: X_train, y: Y_train}
        valid_dict = {x: X_valid, y: Y_valid}

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

            X_shuffled, Y_shuffled = shuffle_data(X_train, Y_train)
            for batch in range(int(X_train.shape[0] / batch_size + 1)):
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

        epoch_train_cost = sess.run(loss, feed_dict=train_dict)
        epoch_train_acc = sess.run(accuracy, feed_dict=train_dict)
        epoch_valid_cost = sess.run(loss, feed_dict=valid_dict)
        epoch_valid_acc = sess.run(accuracy, feed_dict=valid_dict)
        print("After {} epochs:".format(epochs))
        print("\tTraining Cost: {}".format(epoch_train_cost))
        print("\tTraining Accuracy: {}".format(epoch_train_acc))
        print("\tValidation Cost: {}".format(epoch_valid_cost))
        print("\tValidation Accuracy: {}".format(epoch_valid_acc))
        saver.save(sess, save_path)

    return save_path
