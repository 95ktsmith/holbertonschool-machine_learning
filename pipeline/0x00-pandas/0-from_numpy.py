#!/usr/bin/env python3
""" From numpy """
import pandas as pd


def from_numpy(array):
    """
    Creates a Dataframe from a np.ndarray
    array: The np.ndarray to create the dataframe from
    Returns: the created Dataframe
    """
    df = pd.DataFrame(
        array,
        index=[i for i in range(array.shape[0])],
        columns=[chr(i + 65) for i in range(array.shape[1])]
    )

    return df
