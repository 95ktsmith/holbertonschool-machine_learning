#!/usr/bin/env python3
""" Brightness """
import tensorflow as tf


def change_brightness(image, max_delta):
    """
    Randomly changes the brightness of an image
    image: 3D tf.Tensor containing the image to change
    max_delta: Maximum amount the image should be brightened or darkened
    Returned the altered the image
    """
    return tf.image.adjust_brightness(image, max_delta)
