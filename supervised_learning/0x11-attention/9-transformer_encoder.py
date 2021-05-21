#!/usr/bin/env python3
""" Transformer Encoder """
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """
    Class representation of an encoder for a transformer
    """
    def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len,
                 drop_rate=0.1):
        """
        N: Number of blocks in the encoder
        dm: Dimensionality of the model
        h: Number of heads
        hidden: Number of hidden units in the fully connected layer
        input_vocab: Size of the input vocabulary
        max_seq_len: Maximum sequence length possible
        drop_rate: Dropout rate
        """
        super(Encoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = []
        for n in range(N):
            self.blocks.append(EncoderBlock(dm, h, hidden, drop_rate))
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
        x: tensor of shape (batch, input_seq_len, dm) containing the input to
           the encoder
        training: boolean to determine if the model is training
        mask: mask to the applied for multi head attention
        Returns: tensor of shape (batch, input_seq_len, dm) containing the
                 encoder output
        """
        seq_len = int(x.shape[1])

        # Pass input through embedding layer
        x = self.embedding(x)

        # Add positional encoding, pass through dropout layer
        x *= tf.math.sqrt(tf.cast(self.dm, 'float32'))
        x += self.positional_encoding[:seq_len]
        x = self.dropout(x, training=training)

        # Pass through each encoding block
        for block in self.blocks:
            x = block(x, training, mask)

        return x
