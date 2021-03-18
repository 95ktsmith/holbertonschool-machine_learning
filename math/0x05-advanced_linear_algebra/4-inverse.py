#!/usr/bin/env python3
""" Inverse """


def determinant(matrix):
    """
    Calculates the determinant of a matrix
    matrix must be a square list of lists
    Returns the determinant of the matrix
    """
    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    for e in matrix:
        if type(e) is not list:
            raise TypeError("matrix must be a list of lists")
        if len(e) != len(matrix):
            if not (len(e) == 0 and len(matrix) == 1):
                raise ValueError("matrix must be a square matrix")

    if len(matrix) == 1 and len(matrix[0]) == 0:
        return 1
    if len(matrix) == 1 and len(matrix[0]):
        return matrix[0][0]
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    sign = 1
    for col in range(len(matrix)):
        subMatrix = []
        for row in matrix[1:]:
            subMatrix.append(row[:col] + row[col + 1:])
        det += matrix[0][col] * determinant(subMatrix) * sign
        sign *= -1
    return det


def minor(matrix):
    """
    Calculates the matrix of minors of a matrix
    matrix must be a non-empty square list of lists
    Returns the matrix of minors of matrix
    """
    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    for e in matrix:
        if type(e) is not list:
            raise TypeError("matrix must be a list of lists")
        if len(e) != len(matrix):
            raise ValueError("matrix must be a non-empty square matrix")

    size = len(matrix)
    if size == 1:
        return [[1]]

    minors = []
    for row in range(size):
        minors.append([])
        for col in range(size):
            subMatrix = []
            for r in range(size):
                if r != row:
                    subMatrix.append(matrix[r][:col] + matrix[r][col + 1:])
            minors[row].append(determinant(subMatrix))
    return minors


def cofactor(matrix):
    """
    Calculates the cofactor of a matrix
    matrix must be a non-empty square list of lists
    Returns the cofactor of matrix
    """
    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    for e in matrix:
        if type(e) is not list:
            raise TypeError("matrix must be a list of lists")
        if len(e) != len(matrix):
            raise ValueError("matrix must be a non-empty square matrix")

    minorMatrix = minor(matrix)
    cofactorMatrix = []
    for row in range(len(matrix)):
        cofactorMatrix.append([])
        for col in range(len(matrix)):
            if row % 2 == 0 and col % 2 == 0:
                cofactorMatrix[row].append(minorMatrix[row][col])
            elif row % 2 == 1 and col % 2 == 1:
                cofactorMatrix[row].append(minorMatrix[row][col])
            else:
                cofactorMatrix[row].append(minorMatrix[row][col] * -1)

    return cofactorMatrix


def adjugate(matrix):
    """
    Calculates the adjugate of a matrix
    matrix must be a non-empty square list of lists
    Returns the adjugate of matrix
    """
    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    for e in matrix:
        if type(e) is not list:
            raise TypeError("matrix must be a list of lists")
        if len(e) != len(matrix):
            raise ValueError("matrix must be a non-empty square matrix")

    adjugateMatrix = [[] for i in range(len(matrix))]
    cofactorMatrix = cofactor(matrix)
    for row in range(len(matrix)):
        for col in range(len(matrix)):
            adjugateMatrix[col].append(cofactorMatrix[row][col])
    return adjugateMatrix


def inverse(matrix):
    """
    Calculates the inverse of a matrix
    matrix should be a non-empty square matrix
    Returns the inverse of matrix, or None if matrix is singular
    """
    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    for e in matrix:
        if type(e) is not list:
            raise TypeError("matrix must be a list of lists")
        if len(e) != len(matrix):
            raise ValueError("matrix must be a non-empty square matrix")

    det = determinant(matrix)
    if det == 0:
        return None

    adjugateMatrix = adjugate(matrix)
    inverseMatrix = []
    for row in range(len(matrix)):
        inverseMatrix.append([])
        for col in range(len(matrix)):
            inverseMatrix[row].append(adjugateMatrix[row][col] / det)
    return inverseMatrix
