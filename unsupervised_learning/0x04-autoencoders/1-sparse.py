#!/usr/bin/env python3
""" Spare Autoencoder """
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    Creates an autoencoder
    Returns encoder, decoder, auto
        encoder is the encoder model
        decoder is the decoder model
        auto is the fuller autoencoder model
    """
    # Build encoder
    enc_input = keras.Input(shape=(input_dims,))
    enc_layer = keras.layers.Dense(
        hidden_layers[0],
        activation='relu',
    )(enc_input)
    for i in range(1, len(hidden_layers)):
        enc_layer = keras.layers.Dense(
            hidden_layers[i],
            activation='relu',
        )(enc_layer)

    # Final encoder layer
    enc_output = keras.layers.Dense(
        latent_dims,
        activation='relu',
        activity_regularizer=keras.regularizers.l1(lambtha)
    )(enc_layer)

    # Encoder model
    encoder = keras.Model(inputs=enc_input, outputs=enc_output)

    # Build decoder
    dec_input = keras.Input(shape=(latent_dims,))
    dec_layer = keras.layers.Dense(
        hidden_layers[-1],
        activation='relu',
    )(dec_input)
    for i in range(len(hidden_layers) - 2, -1, -1):
        dec_layer = keras.layers.Dense(
            hidden_layers[i],
            activation='relu',
        )(dec_layer)

    # Final decoder layer
    dec_output = keras.layers.Dense(
        input_dims,
        activation='sigmoid',
    )(dec_layer)

    # Decoder model
    decoder = keras.Model(inputs=dec_input, outputs=dec_output)

    # Combine into full autoencoder
    auto = keras.Model(enc_input, decoder(encoder(enc_input)))
    auto.compile(
        optimizer=keras.optimizers.Adam(),
        loss="binary_crossentropy"
    )

    return encoder, decoder, auto
