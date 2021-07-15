#!/usr/bin/env python3
""" Convolutional Autoencoder """
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates an autoencoder
    Returns encoder, decoder, auto
        encoder is the encoder model
        decoder is the decoder model
        auto is the fuller autoencoder model
    """
    # Build encoder
    enc_input = keras.Input(shape=input_dims)
    enc = keras.layers.Conv2D(
        filters=filters[0],
        kernel_size=3,
        padding='same',
        activation='relu'
    )(enc_input)
    enc = keras.layers.MaxPooling2D(
        pool_size=2,
        padding='same'
    )(enc)
    for i in range(1, len(filters)):
        enc = keras.layers.Conv2D(
            filters=filters[i],
            kernel_size=3,
            padding='same',
            activation='relu'
        )(enc)
        enc = keras.layers.MaxPooling2D(
            pool_size=2,
            padding='same'
        )(enc)

    # Encoder model
    encoder = keras.Model(inputs=enc_input, outputs=enc)

    # Build decoder
    dec_input = keras.Input(shape=latent_dims)
    dec = keras.layers.Conv2D(
        filters=filters[-1],
        kernel_size=3,
        padding='same',
        activation='relu'
    )(dec_input)
    dec = keras.layers.UpSampling2D(size=(2, 2))(dec)
    for i in range(len(filters) - 2, -1, -1):
        dec = keras.layers.Conv2D(
            filters=filters[i],
            kernel_size=3,
            padding='valid' if i == 0 else 'same',
            activation='relu'
        )(dec)
        dec = keras.layers.UpSampling2D(size=(2, 2))(dec)

    # Final decoder layer
    dec_output = keras.layers.Conv2D(
        filters=input_dims[-1],
        kernel_size=3,
        activation='sigmoid',
        padding='same'
    )(dec)

    # Decoder model
    decoder = keras.Model(inputs=dec_input, outputs=dec_output)

    # Combine into full autoencoder
    auto = keras.Model(enc_input, decoder(encoder(enc_input)))
    auto.compile(
        optimizer=keras.optimizers.Adam(),
        loss="binary_crossentropy"
    )

    return encoder, decoder, auto
