#!/usr/bin/env python3
""" RNN Encoder """
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """
    Class representation of an RNN Encoder for machine translation
    """
    def __init__(self, vocab, embedding, units, batch):
        """
        vocab: integer representing the size of the input vocabulary
        embedding: integer representing the dimensionality of the embedding
                   vector
        units: integer representing the number of hidden units in the RNN cell
        batch: integer representing the batch size
        """
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab,
            output_dim=embedding
        )
        self.gru = tf.keras.layers.GRU(
            units=self.units,
            return_sequences=True,
            return_state=True,
            kernel_initializer="glorot_uniform"
        )

    def initialize_hidden_state(self):
        """
        Initializes the hidden states for the RNN cell to a tensor of zeros
        Returns: a tensor of shape (batch, units) containing the intialized
                 hidden states.
        """
        return tf.keras.initializers.Zeros()(shape=(self.batch, self.units))

    def call(self, x, initial):
        """
        x: tensor of shape (batch, input_seq_len) containing the input to the
           encoder layer as word indices within the vocabulary
        initial: tensor of shape (batch, units) containing the initial hidden
                 state.
        Returns: outputs, hidden
            outputs: tensor for shape (batch, input_seq_len, units) containing
                     the outputs of the encoder
            hidden: tensor of shape (batch, units) containing the hidden state
                    of the encoder
        """
        outputs, hidden = self.gru(
            inputs=self.embedding(x),
            initial_state=initial
        )
        return outputs, hidden
