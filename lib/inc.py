#! /usr/bin/env python3
"""
Authors: fengyukun
Date:   2016-05-19
Brief:  Common headers
"""

import numpy as np
import logging

np.random.seed(1)

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s]%(filename)s:%(lineno)"
    "s[function:%(funcName)s] %(message)s"
)

# Common function

def sigmoid(x):
    """
    The sigmoid function. This is numerically-stable version
    x: float
    """
    if x >= 0:
        return 1 / (1 + np.exp(-x))
    else:
        exp_x = np.exp(x)
        return exp_x / (exp_x + 1)


def sigmoid_array(x):
    """
    Numerically-stable sigmoid function
    x: ndarray (float)
    """
    vfunc = np.vectorize(sigmoid)
    return vfunc(x)


def softmax(x):
    """
    Numerically-stable softmax function
    x: 2d numpy.ndarray
        The input data.
    """
    stable_input = (x - np.max(x, axis=1) .reshape(x.shape[0], 1))
    stable_input_exp = np.exp(stable_input)
    forward_out = (
        stable_input_exp /
        np.sum(stable_input_exp, axis=1).reshape(stable_input_exp.shape[0], 1)
    )

    return forward_out


def get_col_from_jagged_array(pos, jagged_array):
    """
    Get column from jagged array
    pos: int or 1d array
        Column position at array. If pos == -1, the last column will return.
        Units will be not added if pos is invalid at some rows
    jagged_array: 3d array-like
        eg., [[1dndarray, 1dndarray], [...], [...]]
    Return
    -----
    column: numpy.ndarray
    """

    res = []
    for i in range(0, len(jagged_array)):
        row = jagged_array[i]
        item = []
        if len(row) == 0:
            res.append([])
        else:
            if type(pos) == int:
                res.append(row[pos])
            else:
                res.append(row[pos[i]])

    return np.asarray(res)


def add_two_array(left_array, right_array):
    """
    Add two array.
    left_array: 2d array like
        Each row in left_array is 1d numpy array or empty list
    right_array: 2d array like
        Each row in right_array is 1d numpy array or empty list
    Return
    ----
    added_array: 2d numpy array
    """

    res = []
    for left_row, right_row in zip(left_array, right_array):
        if len(left_row) == 0:
            res.append(right_row)
        elif len(right_row) == 0:
            res.append(left_row)
        else:
            res.append(right_row + left_row)
    return np.array(res)


def merge_jagged_array(left_array, right_array):
    """
    (Not tested)Merge left_array and right_array in horizontal direction
    left_array: jagged_array
    right_array: jagged_array
    Return
    ----
    Merged jagged array
    """
    if len(left_array) != len(right_array):
        logging.error("left_array and right_array are not the same length")
        raise Exception
    merged_array = []
    for left_row, right_row in zip(left_array, right_array):
        merged_array.append(left_row + right_row)
    return merged_array


def print_jagged_array(jagged_array):
    """
    Print jagged_array
    jagged_array: 3d array-like
        eg., [[1dndarray, 1dndarray], [...], [...]]
    """

    for row in jagged_array:
        str_row = ""
        for col in row:
            str_row += str(col) + ", "
        print("row[%s]: %s" % (len(row), str_row.strip(', ')))


def make_jagged_array(n_row, min_col, max_col, max_int, min_int=0,
                      dim_unit=None):
    """
    Make jagged array
    n_row: int
        The number of row in jagged_array
    min_col: int
        The min number of column in jagged_array
    max_col: int
        The max number of column in jagged_array
    max_int: int
        The max interger in jagged_array
    min_int: int
        The minimal interger in jagged_array
    dim_unit: int
        The dimension in each unit.
        None: 2d jagged_array
        Other valid number: 3d jagged_array
    """

    x = []
    for i in range(0, n_row):
        # Random column
        col = np.random.randint(low=min_col, high=max_col + 1)
        x_row = []
        for j in range(0, col):
            if dim_unit is None:
                x_row.append(np.random.randint(low=min_int, high=max_int))
            else:
                x_row.append(np.random.randint(low=min_int, high=max_int,
                                               size=(dim_unit, )))
        x.append(x_row)
    return x


def split_jagged_array(jagged_array, split_pos=None):
    """
    Split jagged array at split_pos
    jagged_array: 3d array-like
        eg., [[1dndarray, 1dndarray], [...], [...]]
    split_pos: 1d array like
        The position array to split. If it is None, middle split will be
        returned
    Return
    ------
    left_array: jagged_array
    right_array: jagged_array
    """

    # if split_pos is None:
        # logging.error("Split position is None")
        # raise Exception
    if split_pos is None:
        split_pos = np.zeros(shape=len(jagged_array), dtype=np.int64)
        for i in range(0, len(jagged_array)):
            split_pos[i] = int(len(jagged_array[i]) / 2)
    left_array = []
    right_array = []
    for i in range(0, len(jagged_array)):
        row = jagged_array[i]
        left_array.append(row[0:split_pos[i]])
        right_array.append(row[split_pos[i]:])
    return (left_array, right_array)


def inverse_jagged_array(jagged_array):
    """
    Inverse jagged array.
    jagged_array: 3d array-like
        eg., [[1dndarray, 1dndarray], [...], [...]]
    Return
    ------
    Inversed jagged array
    """

    inversed_jagged_array = []
    for row in jagged_array:
        inversed_jagged_array.append(list(reversed(row)))

    return inversed_jagged_array

def set_jagged_array(jagged_array, val):
    """set jagged array with val

    :jagged_array: 3d array
    :val: int, the value to be set
    :returns: None

    """
    for i in range(0, len(jagged_array)):
        for j in range(0, len(jagged_array[i])):
            for k in range(0, len(jagged_array[i][j])):
                jagged_array[i][j][k] = val


def jagged_array_test():
    n_row = 5
    min_col = 1
    max_col = 4
    dim_unit = 4
    max_int = 10
    x = make_jagged_array(n_row=n_row, min_col=min_col, max_col=max_col,
                          max_int=max_int, min_int=0, dim_unit=dim_unit)
    print("x:")
    print_jagged_array(x)
    print(get_col_from_jagged_array(-1, x))

    left_array, right_array = split_jagged_array(x)
    print("left_array:")
    print_jagged_array(left_array)
    print("right_array:")
    print_jagged_array(right_array)
    print("inverse x")
    print_jagged_array(inverse_jagged_array(x))
    val = 0
    print("set jagged array with %s" % val)
    set_jagged_array(x, val)
    print_jagged_array(x)


if __name__ == "__main__":
    jagged_array_test()
