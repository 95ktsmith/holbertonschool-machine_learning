#!/usr/bin/env python3
""" Self Attention """
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """
    Class to calculate the attention for machine translation
    """
    def __init__(self, units):
        """
        units: an integer representing the number of hidden units in the
               alignment model
        """
        super(SelfAttention, self).__init__()
        self.W = tf.layers.Dense(units)
        self.U = tf.layers.Dense(units)
        self.V = tf.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """
        s_prev: tensor of shape (batch, units) containing the previous decoder
                hidden state
        hidden_states: tensor of shape *(batch, input_seq_len, units)
                       containing the outputs of the encoder
        Returns: context, weights
                 context: tensor of shape (batch, units) that contains the
                          context vector for the decoder
                 weights: tensor of shape (batch, input_seq_len, 1) that
                          contains the attention weights
        """
        # Reshape previous state
        S = tf.expand_dims(s_prev, axis=1)

        # Attention weights
        A = self.V(tf.nn.tanh(self.W(S) + self.U(hidden_states)))
        weights = tf.nn.softmax(A, axis=1)

        # Context
        context = tf.reduce_sum(weights * hidden_states, axis=1)

        return context, weights
