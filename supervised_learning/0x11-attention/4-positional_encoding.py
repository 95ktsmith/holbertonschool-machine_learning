#!/usr/bin/env python3
""" Positional Encoding """
import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    Calculates the positional encoding for a transformer
    max_seq_len: integer representing the maximum sequence length
    dm: integer representing the model depth
    Returns: numpy.ndarray of shape (max_seq_len, dm) containing the positional
             encoding vectors
    """
    PE = np.zeros((max_seq_len, dm))
    for row in range(max_seq_len):
        for col in range(0, dm, 2):
            PE[row, col] = np.sin(row / (10000 ** (col / dm)))
            PE[row, col + 1] = np.cos(row / (10000 ** (col / dm)))
    return PE
