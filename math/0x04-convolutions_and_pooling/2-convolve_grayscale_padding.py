#!/usr/bin/env python3
""" Convolve with custom padding """
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs a convolution on grayscale images with custom padding
    images is a numpy.ndarray with shape (m, h, w) containing multiple
        grayscale images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
    kernel is a numpy.ndarray with shape (kh, kw) containing the kernel for the
        convolution
        kh is the height of the kernel
        kw is the width of the kernel
    padding is a tuple of (ph, pw)
        ph is the padding for the height of the image
        pw is the padding for the width of the image
    Returns: a numpy.ndarray containing the convolved images
    """
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    ih = images.shape[1]
    iw = images.shape[2]
    ph = padding[0]
    pw = padding[1]
    padded = np.pad(images,
                    ((0, 0), (ph, ph), (pw, pw)),
                    'constant')
    ch = padded.shape[1] - kh + 1
    cw = padded.shape[2] - kw + 1
    convolved = np.zeros((images.shape[0], ch, cw))
    for row in range(ih):
        for col in range(iw):
            masked = padded[:, row:row + kh, col:col + kw] * kernel[None, :, :]
            convolved[:, row, col] = np.sum(masked, axis=(1, 2))
    return convolved
