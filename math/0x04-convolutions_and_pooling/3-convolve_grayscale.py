#!/usr/bin/env python3
""" Convolve with stride """
import numpy as np
from math import floor


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on grayscale images
    images is a numpy.ndarray with shape (m, h, w) containing multiple
        grayscale images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
    kernel is a numpy.ndarray with shape (kh, kw) containing the kernel for the
        convolution
        kh is the height of the kernel
        kw is the width of the kernel
    padding is either a tuple of (ph, pw), ‘same’, or ‘valid’
            if ‘same’, performs a same convolution
            if ‘valid’, performs a valid convolution
        if a tuple:
            ph is the padding for the height of the image
            pw is the padding for the width of the image
    stride is a tuple of (sh, sw)
        sh is the stride for the height of the image
        sw is the stride for the width of the image
    Returns: a numpy.ndarray containing the convolved images
    """
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    ih = images.shape[1]
    iw = images.shape[2]

    if type(padding) is tuple:
        ph = padding[0]
        pw = padding[1]
    elif padding == "valid":
        ph = 0
        pw = 0
    else:  # padding == "same"
        ph = floor((kh - 1) / 2)
        pw = floor((kw - 1) / 2)

    padded = np.pad(images,
                    ((0, 0), (ph, ph), (pw, pw)),
                    'constant')
    sh = stride[0]
    sw = stride[1]
    ch = floor((padded.shape[1] - kh + 1) / sh)
    cw = floor((padded.shape[2] - kw + 1) / sw)
    convolved = np.zeros((images.shape[0], ch, cw))

    for row in range(ch):
        for col in range(cw):
            if row * sh + kh >= padded.shape[1]:
                continue
            if col * sw + kw >= padded.shape[2]:
                continue
            masked = padded[:, row*sh:row*sh + kh, col*sw:col*sw + kw] *\
                kernel[None, :, :]
            convolved[:, row, col] = np.sum(masked, axis=(1, 2))

    return convolved