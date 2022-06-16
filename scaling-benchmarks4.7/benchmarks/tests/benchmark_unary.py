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
import warnings

import numpy as np
import scipy as sc
import qutip as qt


def get_expm(dtype):
    if dtype == qt.Qobj:
        def expm(A, density, rep):
            for _ in range(rep):
                x = A.expm(method=density)

            return x
    else:
        if dtype == sc:
            op = sc.sparse.linalg.expm
        
        elif dtype == np:
            op = sc.linalg.expm
    
        def expm(A, density, rep):
            for _ in range(rep):
                x = op(A)

            return x

    

    return expm


def get_eigenvalues(dtype):
    if dtype == qt.Qobj:
        def eigenvalues(A, density, rep):
            issparse = "sparse" == density
            for _ in range(rep):
                x = A.eigenenergies(sparse=issparse)
            return x

    else:   
        if dtype == np:
            op = np.linalg.eigvalsh
        elif dtype == sc:
            # Omit in benchmarks
            raise NotImplementedError
    

        def eigenvalues(A, density, rep):
            for _ in range(rep):
                x = op(A)

            return x
    return eigenvalues
