#!/usr/bin/env python3
""" Crop image """
import tensorflow as tf


def crop_image(image, size):
    """
    Performs a random crop of an image
    image: 3D tf.Tensor containing the image to crop
    size: tuple containing the size of the crop
    """
    return tf.image.random_crop(image, size)
