"""This file contains the unary operations that will be benchmarked. All the functions
in this file should return the function that will be tested. The tested funtion must
have as inputs:

    A: input 2D array

    dtype: data type of A (np, tf, qt.data.Dense ... )

    rep: number of times that the operations will be repeated.
The function does not need to return anything else. The getters have as input parameters
only dtype. If the getter returns a NotImplementedError it will be omitted in the
benchmarks.
    """
import qutip as qt

def get_matmul():
    def matmul(op, ket, rep):
        for _ in range(rep):
            x = op@ket
        return x

    return matmul


def get_expect_matmul():
    def ex_matmul(op, ket, rep):
        for _ in range(rep):
            x = op(2)@ket

        return x
    return ex_matmul
