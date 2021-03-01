#!/usr/bin/env python3
""" Neural Style Transfer """
import numpy as np
import tensorflow as tf


class NST:
    """
    Class to perform tasks for neural style transfer
    """
    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ]
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        style_image - the image used as a style reference, stored as a
            numpy.ndarray
        content_image - the image used as a content reference, stored as a
            numpy.ndarray
        alpha - the weight for content cost
        beta - the weight for style cost
        """
        if type(style_image) is not np.ndarray or \
                len(style_image.shape) != 3 or \
                int(style_image.shape[2]) != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )
        if type(content_image) is not np.ndarray or \
                len(content_image.shape) != 3 or \
                int(content_image.shape[2]) != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)"
            )
        if type(alpha) not in [int, float] or alpha <= 0:
            raise TypeError(
                "alpha must be a non-negative number"
            )
        if type(beta) not in [int, float] or beta <= 0:
            raise TypeError(
                "beta must be a non-negative number"
            )
        tf.enable_eager_execution()

        self.style_image = NST.scale_image(style_image)
        self.content_image = NST.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        """
        Rescales an image such that its pixels' values are between
        0 and 1 and its largest side is 512 pixels
        """
        if type(image) is not np.ndarray or \
                len(image.shape) != 3 or \
                int(image.shape[2]) != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )
        h, w = image.shape[0], image.shape[1]
        image = tf.expand_dims(image, 0)
        if h > w:
            image = tf.image.resize_bicubic(
                image,
                (512, int(w * 512 / h)),
            )
        else:
            image = tf.image.resize_bicubic(
                image,
                (int(h * 512 / w), 512)
            )

        image = image / 255
        image = tf.clip_by_value(image, 0, 1)
        return image
