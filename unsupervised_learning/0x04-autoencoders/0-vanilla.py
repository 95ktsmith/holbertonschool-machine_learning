#!/usr/bin/env python3
import tensorflow.keras as K


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates an autoencoder
    Returns encoder, decoder, auto
        encoder is the encoder model
        decoder is the decoder model
        auto is the fuller autoencoder model
    """
    # Build encoder
    enc_input = K.Input(shape=(input_dims,))
    enc_layer = K.layers.Dense(hidden_layers[0], activation='relu')(enc_input)
    for i in range(1, len(hidden_layers)):
        enc_layer = K.layers.Dense(
            hidden_layers[i],
            activation='relu'
        )(enc_layer)

    # Final encoder layer
    enc_output = K.layers.Dense(
        latent_dims,
        activation='relu'
    )(enc_layer)

    # Encoder model
    encoder = K.Model(inputs=enc_input, outputs=enc_output)

    # Build decoder
    dec_input = K.Input(shape=(latent_dims,))
    dec_layer = K.layers.Dense(hidden_layers[-1], activation='relu')(dec_input)
    for i in range(len(hidden_layers) - 2, -1, -1):
        dec_layer = K.layers.Dense(
            hidden_layers[i],
            activation='relu'
        )(dec_layer)

    # Final decoder layer
    dec_output = K.layers.Dense(
        input_dims,
        activation='sigmoid'
    )(dec_layer)

    # Decoder model
    decoder = K.Model(inputs=dec_input, outputs=dec_output)

    # Combine into full autoencoder
    auto = K.Sequential()
    auto.add(encoder)
    auto.add(decoder)
    auto.compile(
        optimizer=K.optimizers.Adam(),
        loss="binary_crossentropy"
    )

    return encoder, decoder, auto
