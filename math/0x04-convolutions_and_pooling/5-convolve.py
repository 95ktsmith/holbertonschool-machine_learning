#!/usr/bin/env python3
""" Convolution with multiple kernels """
import numpy as np
from math import floor


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images with channels
    images is numpy.ndarray with shape (m, h, w, c) containing multiple images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
        c is the number of channels in the image
    kernel is a numpy.ndarray with shape (kh, kw, c, nc) containing the kernel
        for the convolution
        kh is the height of the kernel
        kw is the width of the kernel
        nc is the number of kernels
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
    kh = kernels.shape[0]
    kw = kernels.shape[1]
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
                    ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                    'constant')
    sh = stride[0]
    sw = stride[1]
    ch = floor((padded.shape[1] - kh + 1) / sh)
    cw = floor((padded.shape[2] - kw + 1) / sw)
    convolved = np.zeros((images.shape[0], ch, cw, kernels.shape[3]))

    for k in range(kernels.shape[3]):
        for row in range(ch):
            for col in range(cw):
                if row * sh + kh >= padded.shape[1]:
                    continue
                if col * sw + kw >= padded.shape[2]:
                    continue
                mask = padded[:, row*sh:row*sh + kh, col*sw:col*sw + kw, :] *\
                    kernels[None, :, :, :, k]
                convolved[:, row, col, k] = np.sum(mask, axis=(1, 2, 3))

    return convolved
