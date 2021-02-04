#!/usr/bin/env python3
""" Pool backwards propagation """
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs back propagation over a pooling layer of a neural network
    dA is a numpy.ndarray of shape (m, h_new, w_new, c_new) containing the
        partial derivatives with respect to the output of the pooling layer
        m is the number of examples
        h_new is the height of the output
        w_new is the width of the output
        c is the number of channels
    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c) containing the
        output of the previous layer
        h_prev is the height of the previous layer
        w_prev is the width of the previous layer
    kernel_shape is a tuple of (kh, kw) containing the size of the kernel for
        the pooling
        kh is the kernel height
        kw is the kernel width
    stride is a tuple of (sh, sw) containing the strides for the pooling
        sh is the stride for the height
        sw is the stride for the width
    mode is a string containing either max or avg, indicating whether to
        perform maximum or average pooling, respectively
    Returns: the partial derivatives with respect to the previous layer
    """
    m, h_new, w_new, c_new = dA.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros(A_prev.shape)
    for img in range(m):
        for channel in range(c_new):
            for row in range(h_new):
                for col in range(w_new):
                    if mode == "max":
                        pool = A_prev[img,
                                      row * sh:row * sh + kh,
                                      col * sw:col * sw + kw,
                                      channel]
                        mask = pool == np.max(pool)
                        dA_prev[img,
                                row * sh:row * sh + kh,
                                col * sw:col * sw + kw,
                                channel] += dA[img, row, col, channel] * mask
                    else:  # mode == avg
                        d = dA[img, row, col, channel] / (kh * kw)
                        dA_prev[img,
                                row * sh:row * sh + kh,
                                col * sw:col * sw + kw,
                                channel] += d
    return dA_prev
