#!/usr/bin/env python3
""" Transformer Encoder Block """
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """
    Class representation of an encoder block for a transformer
    """
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        dm: Dimensionality of the model
        h: Number of heads
        hidden: Number of hidden units in the fully connected layer
        drop_rate: Dropout rate
        """
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(
            units=hidden,
            activation='relu'
        )
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """
        x: tensor of shape (batch, input_seq_len, dm) containing the input to
           the encoder block
        training: boolean to determine if the model is training
        mask: the mask to be applied for multi head attention
        Returns: tensor of the shape (batch, input_seq_len, dm) containing the
                 block's output
        """
        # Pass through multi head attention and dropout layers
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)

        # Add and normalize
        out1 = self.layernorm1(x + attn_output)

        # Feed through dense layers and dropout layer
        dense_output = self.dense_hidden(out1)
        dense_output = self.dense_output(dense_output)
        dense_output = self.dropout2(dense_output, training=training)

        # Add and normalize
        out2 = self.layernorm2(out1 + dense_output)

        return out2
