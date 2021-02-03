#!/usr/bin/env python3
""" Pooling forward propagation """
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over a pooling layer of a neural network
    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing
        the output of the previous layer
        m is the number of examples
        h_prev is the height of the previous layer
        w_prev is the width of the previous layer
        c_prev is the number of channels in the previous layer
    kernel_shape is a tuple of (kh, kw) containing the size of the kernel for
        the pooling
        kh is the kernel height
        kw is the kernel width
    stride is a tuple of (sh, sw) containing the strides for the pooling
        sh is the stride for the height
        sw is the stride for the width
    mode is a string containing either max or avg, indicating whether to
        perform maximum or average pooling, respectively
    Returns: the output of the pooling layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh = kernel_shape[0]
    kw = kernel_shape[1]
    sh = stride[0]
    sw = stride[1]

    ch = int(((h_prev - kh) / sh) + 1)
    cw = int(((w_prev - kw) / sw) + 1)
    convolved = np.zeros((m, ch, cw, c_prev))

    for row in range(ch):
        for col in range(cw):
            if mode == "avg":
                pooled = np.average(A_prev[:, row*sh:row*sh + kh,
                                    col*sw:col*sw + kw, :],
                                    axis=(1, 2))
            else:  # mode == "max"
                pooled = np.amax(A_prev[:, row*sh:row*sh + kh,
                                 col*sw:col*sw + kw, :],
                                 axis=(1, 2))
            convolved[:, row, col, :] = pooled

    return convolved
