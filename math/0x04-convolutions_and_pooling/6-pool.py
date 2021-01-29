#!/usr/bin/env python3
""" Pooling """
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Performs pooling on images
    images is a numpy.ndarray with shape (m, h, w, c) containing multiple
        images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
        c is the number of channels in the image
    kernel_shape is a tuple of (kh, kw) containing the kernel shape for the
        pooling
        kh is the height of the kernel
        kw is the width of the kernel
    stride is a tuple of (sh, sw)
        sh is the stride for the height of the image
        sw is the stride for the width of the image
    mode indicates the type of pooling
        max indicates max pooling
        avg indicates average pooling
    You are only allowed to use two for loops; any other loops of any kind are
        not allowed
    Returns: a numpy.ndarray containing the pooled images
    """
    kh = kernel_shape[0]
    kw = kernel_shape[1]
    ih = images.shape[1]
    iw = images.shape[2]

    sh = stride[0]
    sw = stride[1]

    ch = int(((padded.shape[1] - kh) / sh) + 1)
    cw = int(((padded.shape[2] - kw) / sw) + 1)
    convolved = np.zeros((images.shape[0], ch, cw, images.shape[3]))

    for row in range(ch):
        for col in range(cw):
            if mode == "average":
                pooled = np.average(images[:, row*sh:row*sh + kh,
                                    col*sw:col*sw + kw, :],
                                    axis=(1, 2))
            else:  # mode == "max"
                pooled = np.amax(images[:, row*sh:row*sh + kh,
                                 col*sw:col*sw + kw, :],
                                 axis=(1, 2))
            convolved[:, row, col, :] = pooled

    return convolved
