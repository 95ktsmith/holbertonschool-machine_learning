#!/usr/bin/env python3
""" Shear image """
import tensorflow as tf


def shear_image(image, intensity):
    """
    Randomly shears an image
    image: 3D tf.Tensor containing the image to shear
    intensity: intensity with which the image should be sheared
    Returns the sheared image
    """
    return tf.keras.preprocessing.image.array_to_img(
        tf.keras.preprocessing.image.random_shear(
            tf.keras.preprocessing.image.img_to_array(image),
            intensity,
            row_axis=0,
            col_axis=1,
            channel_axis=2
        )
    )
