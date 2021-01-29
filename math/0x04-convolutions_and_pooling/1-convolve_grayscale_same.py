#!/usr/bin/env python3
""" Same convolution """
import numpy as np
from math import floor


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images
    images is a numpy.ndarray with shape (m, h, w) containing multiple
        grayscale images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
    kernel is a numpy.ndarray with shape (kh, kw) containing the kernel for the
        convolution
        kh is the height of the kernel
        kw is the width of the kernel
    Returns: a numpy.ndarray containing the convolved images
    """
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    ih = images.shape[1]
    iw = images.shape[2]
    convolved = np.zeros((images.shape[0], ih, iw))
    ph = floor((kh - 1) / 2)
    pw = floor((kw - 1) / 2)
    padded = np.pad(images,
                    ((0, 0), (ph, ph), (pw, pw)),
                    'constant')
    for row in range(ih):
        for col in range(iw):
            masked = padded[:, row:row + kh, col:col + kw] * kernel[None, :, :]
            convolved[:, row, col] = np.sum(masked, axis=(1, 2))
    return convolved
