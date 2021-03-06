#!/usr/bin/env python3
""" Variational Encoder """
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder
    Returns encoder, decoder, auto
        encoder is the encoder model
        decoder is the decoder model
        auto is the fuller autoencoder model
    """
    # Encoder
    enc_input = keras.Input(shape=(input_dims,))
    enc = keras.layers.Dense(
        units=hidden_layers[0],
        activation='relu'
    )(enc_input)
    for i in range(1, len(hidden_layers)):
        enc = keras.layers.Dense(
            units=hidden_layers[i],
            activation='relu'
        )(enc)
    mu = keras.layers.Dense(units=latent_dims)(enc)
    sigma = keras.layers.Dense(units=latent_dims)(enc)

    def sample_z(args):
        """ Sampling function """
        mu, sigma = args
        batch = keras.backend.shape(mu)[0]
        dim = keras.backend.int_shape(mu)[1]
        eps = keras.backend.random_normal(shape=(batch, dim))
        return mu + keras.backend.exp(sigma / 2) * eps

    z = keras.layers.Lambda(sample_z)([mu, sigma])
    encoder = keras.Model(inputs=enc_input, outputs=[mu, sigma, z])

    # Decoder
    dec_input = keras.Input(shape=(latent_dims,))
    dec = keras.layers.Dense(
        units=hidden_layers[-1],
        activation='relu'
    )(dec_input)
    for i in range(len(hidden_layers) - 2, -1, -1):
        dec = keras.layers.Dense(
            units=hidden_layers[i],
            activation='relu'
        )(dec)
    dec_out = keras.layers.Dense(
        units=input_dims,
        activation='sigmoid'
    )(dec)
    decoder = keras.Model(inputs=dec_input, outputs=dec_out)

    def kl_reconstruction_loss(true, pred):
        """
        Loss function combining binary crossentropy and KL divergence
        """
        reconstruction_loss = keras.losses.binary_crossentropy(
            true,
            pred
        ) * input_dims
        kl_loss = 1 + sigma - keras.backend.square(mu)-keras.backend.exp(sigma)
        kl_loss = keras.backend.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        return keras.backend.mean(reconstruction_loss + kl_loss)

    outputs = decoder(encoder(enc_input)[2])
    auto = keras.Model(inputs=enc_input, outputs=outputs)
    auto.compile(
        optimizer='adam',
        loss=kl_reconstruction_loss
    )
    return encoder, decoder, auto
