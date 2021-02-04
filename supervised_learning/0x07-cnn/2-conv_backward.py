#!/usr/bin/env python3
""" Convolution back propagation """
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs back propagation over a convolutional layer of a neural network
    dZ is a numpy.ndarray of shape (m, h_new, w_new, c_new) containing the
        partial derivatives with respect to the unactivated output of the
        convolutional layer
        m is the number of examples
        h_new is the height of the output
        w_new is the width of the output
        c_new is the number of channels in the output
    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing
        the output of the previous layer
        h_prev is the height of the previous layer
        w_prev is the width of the previous layer
        c_prev is the number of channels in the previous layer
    W is a numpy.ndarray of shape (kh, kw, c_prev, c_new) containing the
        kernels for the convolution
        kh is the filter height
        kw is the filter width
    b is a numpy.ndarray of shape (1, 1, 1, c_new) containing the biases
        applied to the convolution
    padding is a string that is either same or valid, indicating the type of
        padding used
    stride is a tuple of (sh, sw) containing the strides for the convolution
        sh is the stride for the height
        sw is the stride for the width
    Returns: The partial derivatives with respect to the previous layer, the
        kernels, and the biases, respectively
    """
    m, h_new, w_new, c_new = dZ.shape
    h_prev, w_prev, c_prev = A_prev.shape[1:]
    kh, kw = W.shape[:2]
    sh, sw = stride

    if padding == "valid":
        ph = 0
        pw = 0
    else:  # padding == "same"
        ph = int((ch * sh - h_prev + kh - 1) / 2)
        pw = int((cw * sw - w_prev + kw - 1) / 2)

    padded = np.pad(A_prev,
                    ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                    'constant',
                    constant_values=0)
    dA_prev = np.zeros(padded.shape)
    dW = np.zeros(W.shape)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for img in range(m):
        for channel in range(c_new):
            for row in range(h_new):
                for col in range(w_new):
                    grad = W[:, :, :, channel] * dZ[img, row, col, channel]
                    dA_prev[img,
                            row * sh:row * sh + kh,
                            col * sw:col * sw + kw,
                            :] += grad

                    grad = padded[img,
                                  row * sh:row * sh + kh,
                                  col * sw:col * sw + kw,
                                  :] * dZ[img, row, col, channel]
                    dW[:, :, :, channel] += grad

    dA_prev = dA_prev[:, ph:dA_prev.shape[1] - ph, pw:dA_prev.shape[2] - pw, :]

    return dA_prev, dW, db
