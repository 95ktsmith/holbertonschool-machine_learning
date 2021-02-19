#!/usr/bin/env python3
""" Transfer Learning """
import tensorflow.keras as K


def preprocess_data(X, Y):
    """
    Data pre-processor
    X is a numpy.ndarray of shape (m, 32, 32, 3) containing the CIFAR 10 data,
        where m is the number of data points
    Y is a numpy.ndarray of shape (m,) containing the CIFAR 10 labels for X
    Returns: X_p, Y_p
        X_p is a numpy.ndarray containing the preprocessed X
        Y_p is a numpy.ndarray containing the preprocessed Y
    """
    # Preprocessor used for densenet models
    X_p = K.applications.densenet.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, num_classes=10)
    return X_p, Y_p


if __name__ == "__main__":
    # Import cifar10 data
    train_data, valid_data = K.datasets.cifar10.load_data()

    # Preprocess
    train_X, train_Y = preprocess_data(train_data[0], train_data[1])
    valid_X, valid_Y = preprocess_data(valid_data[0], valid_data[1])

    # Load densenet121 model, excluding fully connected and dense layers
    densenet_model = K.applications.DenseNet121(
        include_top=False,
        weights='imagenet',
        input_shape=(32, 32, 3)
    )

    # Build full model, including new fully connected layers after
    # the densenet blocks
    model = K.Sequential()
    model.add(K.layers.InputLayer(input_shape=(32, 32, 3)))
    model.add(densenet_model)
    model.add(K.layers.Flatten())
    model.add(K.layers.Dense(
        units=1024,
        activation='relu',
    ))
    model.add(K.layers.Dropout(rate=0.25))
    model.add(K.layers.Dense(
        units=512,
        activation='relu',
    ))
    model.add(K.layers.Dropout(rate=0.25))
    model.add(K.layers.Dense(
        units=256,
        activation='relu'
    ))
    model.add(K.layers.Dropout(rate=0.25))
    model.add(K.layers.Dense(
        units=128,
        activation='relu'
    ))
    model.add(K.layers.Dropout(rate=0.25))
    model.add(K.layers.Dense(
        units=10,
        activation='softmax'
    ))

    model.compile(
        optimizer="Adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Train
    history = model.fit(
      x=train_X,
      y=train_Y,
      epochs=10,
      verbose=True,
      batch_size=256,
      validation_data=(valid_X, valid_Y)
    )

    # Save model to file
    model.save("cifar10.h5")
