#!/usr/bin/env python3
""" Transformer Decoder """
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """
    Class representation of a decoder for a transformer
    """
    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len,
                 drop_rate=0.1):
        """
        dm - the dimensionality of the model
        h - the number of heads
        hidden - the number of hidden units in the fully connected layer
        target_vocab - the size of the target vocabulary
        max_seq_len - the maximum sequence length possible
        drop_rate - the dropout rate
        """
        super(Decoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = []
        for n in range(N):
            self.blocks.append(DecoderBlock(dm, h, hidden, drop_rate))
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        x - a tensor of shape (batch, target_seq_len, dm) containing the input
            to the decoder
        encoder_output - a tensor of shape (batch, input_seq_len, dm)
            containing the output of the encoder
        training - a boolean to determine if the model is training
        look_ahead_mask - the mask to be applied to the first multi head
            attention layer
        padding_mask - the mask to be applied to the second multi head
            attention layer
        Returns: a tensor of shape (batch, target_seq_len, dm) containing the
            decoder output
        """
        seq_len = int(x.shape[1])

        # Pass through embedding layer
        x = self.embedding(x)

        # Add positional encoding and pass through dropout layer
        x *= tf.math.sqrt(tf.cast(self.dm, 'float32'))
        x += self.positional_encoding[:seq_len]
        x = self.dropout(x, training=training)

        # Pass through each decoder block
        for block in self.blocks:
            x = block(
                x,
                encoder_output,
                training,
                look_ahead_mask,
                padding_mask
            )

        return x
