#!/usr/bin/env python3
""" Create encoder, combined, decoder masks """
import tensorflow.compat.v2 as tf


def create_masks(inputs, target):
    """
    Creates all masks for training/validation
    inputs: tf.Tensor of shape (batch_size, seq_len_in) that contains the input
        sentence
    target: tf.Tensor of shape (batch_size, seq_len_out) that contains the
        target sentence
    Returns: encoder_mask, combined_mask, decoder_mask
        encoder_mask: tf.Tensor padding mask of shape
            (batch_size, 1, 1, seq_len_in) to be applied to the encoder
        combined_mask: tf.Tensor of shape
            (batch_size, 1, seq_len_out, seq_len_out) used in the first
            attention block in the decoder to pad and mask future tokens in the
            input received by the decoder. It takes the maximum between a
            look ahead mask and the decoder target padding mask.
        decoder_mask: tf.Tensor padding mask of shape
            (batch_size, 1, 1, seq_len_in) used in the second attention block
            in the decoder.
    """
    encoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    encoder_mask = encoder_mask[:, tf.newaxis, tf.newaxis, :]
    decoder_mask = tf.identity(encoder_mask)

    decoder_target_mask = tf.cast(tf.math.equal(target, 0), tf.float32)
    decoder_target_mask = decoder_target_mask[:, tf.newaxis, tf.newaxis, :]

    target_seq_len = tf.shape(target)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(
        tf.ones((target_seq_len, target_seq_len)), -1, 0
    )

    combined_mask = tf.maximum(decoder_target_mask, look_ahead_mask)

    return encoder_mask, combined_mask, decoder_mask
