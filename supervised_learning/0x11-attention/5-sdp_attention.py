#!/usr/bin/env python3
""" Scaled Dot-Product Attention """
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
    Q: tensor with shape (..., seq_len_q, dk) containing the query matrix
    K: tensor with shape (..., seq_len_v, dk) containing the key matrix
    V: tensor with shape (..., seq_len_v, dv) containing the value matrix
    mask: tensor that can be broadcast into (..., seq_len_q, seq_len_v)
          containing the optional maask, or defaulted to None
    The Preceding dimensions of Q, K, and V are the same
    Returns: output, weights
             output: tensor with shape (..., seq_len_q, dv) containing the dot
                     product attention
             weights: tensor with shape (..., seq_len_q, seq_len_v) containing
                      the attention weights
    """
    # Matmul Q and K
    QK = tf.matmul(Q, K, transpose_b=True)

    # Scale the dot product
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled = QK / tf.math.sqrt(dk)

    # Add mask if not None
    if mask is not None:
        scaled += mask * -1e9

    # Pass scaled attention through softmax activation
    weights = tf.nn.softmax(scaled, axis=-1)

    # Matmul by value matrix for output
    output = tf.matmul(weights, V)

    return output, weights
