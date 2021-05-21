#!/usr/bin/env python3
""" Transformer Decoder Block """
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """
    Class representation of a decoder block for a transformer
    """
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        dm: Dimensionality of the model
        h: Number of heads
        hidden: Number of hidden units in the fully connected layer
        drop_rate: Dropout rate
        """
        super(DecoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(
            units=hidden,
            activation='relu'
        )
        self.dense_output = tf.keras.layers.Dense(units=dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        x: tensor of shape (batch, target_seq_len, dm)containing the input to
           the decoder block
        encoder_output: tensor of shape (batch, input_seq_len, dm)containing
                        the output of the encoder
        training: boolean to determine if the model is training
        look_ahead_mask: mask to be applied to the first multi head attention
                         layer
        padding_mask: mask to be applied to the second multi head attention
                      layer
        Returns: tensor of shape (batch, target_seq_len, dm) containing the
                 block's output
        """
        # Pass through first round of MHA and dropout layers
        attn_out, _ = self.mha1(x, x, x, look_ahead_mask)
        attn_out = self.dropout1(attn_out, training=training)

        # Add and normalize
        out1 = self.layernorm1(x + attn_out)

        # Pass trough second round of MHA and dropout layers
        attn_out, _ = self.mha2(
            out1,
            encoder_output,
            encoder_output,
            padding_mask
        )
        attn_out = self.dropout2(attn_out, training=training)

        # Add and normalize
        out2 = self.layernorm2(out1 + attn_out)

        # Pass through dense layers and dropout layer
        dense_output = self.dense_hidden(out2)
        dense_output = self.dense_output(dense_output)
        dense_output = self.dropout3(dense_output, training=training)

        # Add and normalize
        out3 = self.layernorm3(out2 + dense_output)

        return out3
