#!/usr/bin/env python3
""" Moving Average """


def moving_average(data, beta):
    """ Calculates the weighted average of a data set, with bias correction
        data is the list of data to calculate the moving average of
        beta is the weight used for the moving average
        Returns: a list containing the moving averages of data
    """
    averages = []
    prev = 0
    for i in range(0, len(data)):
        prev = beta * prev + (1 - beta) * data[i]
        averages.append(prev / (1 - (beta ** (i + 1))))
    return averages
