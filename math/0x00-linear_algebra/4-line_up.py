#!/usr/bin/env python3
""" Add Arrays """


def add_arrays(arr1, arr2):
    """ Returns a new array of arr1 and arr2 added together, or None
        if the two arrays are not the same length
    """
    if len(arr1) != len(arr2):
        return None

    sums = []
    for i in range(0, len(arr1)):
        sums.append(arr1[i] + arr2[i])
    return sums
