#!/usr/bin/env python3
""" RNN Decoder """
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """
    Class representation of a decoder for machine translation
    """
    def __init__(self, vocab, embedding, units, batch):
        """
        vocab: integer representing the size of the output vocabulary
        embedding: integer representing the dimensionality of the embedding
                   vector
        units: integer representing the number of hidden units in the RNN cell
        batch: integer representing the batch size
        """
        super(RNNDecoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab,
            output_dim=embedding
        )
        self.gru = tf.keras.layers.GRU(
            units=units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform"
        )
        self.F = tf.keras.layers.Dense(vocab)

    def call(self, x, s_prev, hidden_states):
        """
        x: tensor of shape (batch, 1) containing the previous word in the
           target sequence as an index of the target vocabulary
        s_prev: tensor of shape (batch, units) containing the previous decoder
                hidden state
        hidden_states: tensor of shape (batch, input_seq_len, units) containing
                       the outputs of the encoder
        Returns: y, s
                 y: tensor of shape (batch, vocab) containing the output word
                    as a one hot vector in the target vocabulary
                 s: tensor of shape (batch, units) containing the new decoder
                    hidden state
        """
        # Get context vector based on previous state and encoder outputs
        context, _ = SelfAttention(s_prev.shape[1])(s_prev, hidden_states)

        # Pass previous word through embedding layer and concatenate to
        # context vector
        embed = self.embedding(x)
        concat = tf.concat([tf.expand_dims(context, 1), embed], axis=-1)

        # Pass through GRU
        output, s = self.gru(concat)

        # Reshape and pass through fully connected layer
        output = tf.reshape(output, (-1, output.shape[2]))
        y = self.F(output)

        return y, s
