#!/usr/bin/env python3
""" Load from file """
import pandas as pd


def from_file(filename, delimiter):
    """
    Loads data from a file as a pd.DataFrame
    filename: the file to load from
    delimiter: the column separator
    Returns: the loaded pd.DataFrame
    """
    df = pd.read_csv(filename, delimiter=delimiter)
    return df
