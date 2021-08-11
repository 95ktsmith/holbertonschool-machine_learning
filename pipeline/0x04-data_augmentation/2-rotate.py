#!/usr/bin/env python3
""" Rotate image """
import tensorflow as tf


def rotate_image(image):
    """
    Rotates an image by 90 degrees counter-clockwise
    image: 3D tf.Tensor containing the image to rotate
    Returns the rotated the image
    """
    return tf.image.rot90(image)
