#!/usr/bin/env python3
""" Convolution forward propagation """
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation over a convolutional layer of a neural network
    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing
        the output of the previous layer
        m is the number of examples
        h_prev is the height of the previous layer
        w_prev is the width of the previous layer
        c_prev is the number of channels in the previous layer
    W is a numpy.ndarray of shape (kh, kw, c_prev, c_new) containing the
        kernels for the convolution
        kh is the filter height
        kw is the filter width
        c_prev is the number of channels in the previous layer
        c_new is the number of channels in the output
    b is a numpy.ndarray of shape (1, 1, 1, c_new) containing the biases
        applied to the convolution
    activation is an activation function applied to the convolution
    padding is a string that is either same or valid, indicating the type of
        padding used
    stride is a tuple of (sh, sw) containing the strides for the convolution
        sh is the stride for the height
        sw is the stride for the width
    Returns: the output of the convolutional layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh = stride[0]
    sw = stride[1]

    if padding == "valid":
        ph = 0
        pw = 0
        ch = int((h_prev - kh) / sh + 1)
        cw = int((w_prev - kw) / sw + 1)
    else:  # padding == "same"
        ch = h_prev
        cw = w_prev
        ph = int((ch * sh - h_prev + kh - 1) / 2)
        pw = int((cw * sw - w_prev + kw - 1) / 2)

    padded = np.pad(A_prev,
                    ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                    'constant',
                    constant_values=0)
    convolved = np.zeros((m, ch, cw, c_new))

    for channel in range(c_new):
        for row in range(ch):
            for col in range(cw):
                mask = padded[:, row*sh:row*sh + kh, col*sw:col*sw + kw, :] *\
                    W[None, :, :, :, channel]
                out = np.sum(mask, axis=(1, 2, 3)) + b[:, :, :, channel]
                convolved[:, row, col, channel] = activation(out)
    return convolved
