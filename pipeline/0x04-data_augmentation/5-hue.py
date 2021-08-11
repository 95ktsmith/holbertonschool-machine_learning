#!/usr/bin/env python3
""" Hue """
import tensorflow as tf


def change_hue(image, delta):
    """
    Changes the hue of an image
    image: 3D tf.Tensor containing the image to change
    delta: Amount the hue should change
    Returns the altered image
    """
    return tf.image.adjust_hue(image, delta)
