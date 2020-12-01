#!/usr/bin/env python3
""" Concat matrices """


def cat_matrices2D(mat1, mat2, axis=0):
    """ Returns a new matrix of mat2 concatenated to ma1
        If axis is 0, mat2 is concatenated as new rows
        If axis is 1, mat2 is concatenated as new columns
        If the two matrices cannot be concateneated (sizes do not line up),
        returns None instead
    """
    # Concat as new rows
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None

        # Create empty matrix with needed number of rows
        cat_mat = [[] for i in range(0, len(mat1) + len(mat2))]

        # Populate new matrix by column, first with mat1 then mat2
        for c in range(0, len(mat1[0])):
            for r in range(0, len(mat1)):
                cat_mat[r].append(mat1[r][c])
            for r in range(0, len(mat2)):
                cat_mat[r + len(mat1)].append(mat2[r][c])
        return cat_mat

    # Concat as new columns
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None

        # Create empty matrix with needed number of rows
        cat_mat = [[] for i in range(0, len(mat1))]

        # Populate new matrix by row, first with mat1 then mat2
        for r in range(0, len(mat1)):
            for c in range(0, len(mat1[r])):
                cat_mat[r].append(mat1[r][c])
            for c in range(0, len(mat2[r])):
                cat_mat[r].append(mat2[r][c])
        return cat_mat

    # else
    return None
