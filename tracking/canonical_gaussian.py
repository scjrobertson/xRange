"""
Module containing the CanonicalGaussian object
and some helper functions.

@author: scj robertson
@since: 22/05/2016
"""

import numpy as np

class CanonicalGaussian:
    '''
    Class for representing a multivariate Gaussian distribution,
    given a partitioned precision matrix and information vector.

    This representation makes use of matrix multiplication
    to align variables and expand scopes. This is computationally
    expensive, but conceptually easy. Not a great representation, but
    I couldn't find an exisiting canonical form representation in python.

    This class doesn't yet handle any value errors or check
    dimensional consistency.

    Parameters
    ----------

    vars_ : list
        A list of variables, integer values representing a variable.
    dims : list
        The respective list of the variables' dimensions.
    info : (d, 1) ndarray
        The information vector.
    prec : (d, d) ndarray
        The precision matrix.
    norm : float
        The normalisation constant.

    Methods
    ----------
    marginalize
        Marginalizes out the given variables and returns a new 
        distribution.
    introduce_evidence
        Sets a subset of given variables into a given state.

    Example
    ----------
    >>> print('To be completed, look at tracking.ipynb for now.')
    '''
    def __init__(self, vars_, dims, info, prec, norm):
        self._vars = list(vars_)
        self._dims = list(dims)
        
        self._info = np.array(info).reshape((sum(dims), 1))
        self._prec = np.array(prec)
        self._norm = norm
        
        if sorted(vars_) != vars_:
            self._order()

    def _order(self):
        '''
        Reorders the arrays so that the variables appear
        in ascending order of their numeric values.
        '''
        v_0, d_0, c_0 = [], [], []
        c_r = np.cumsum(([0] + self._dims[:-1])).tolist()
        N = len(self._vars)

        for i in np.arange(0, N):
            j = np.argmin(self._vars)
            v_0.append(self._vars.pop(j))
            d_0.append(self._dims.pop(j))
            c_0.append(c_r.pop(j))

        r_0 = np.cumsum(([0] + d_0[:-1])).tolist()
        P = block_permutation(r_0, c_0, d_0)

        self._vars = v_0
        self._dims = d_0        
        self._prec = (P)@(self._prec)@(P.T)
        self._info = (P)@(self._info)

    def _rearrange(self, vars_):
        '''
        Moves the given variables to the end of the 
        partition matrix. Precomputing for marginilization
        and introducing evidence.

        Parameters
        ----------
        vars_ : list
            The variables that need to be moved.
        '''
        M = len(vars_)

        c_r = np.cumsum(([0] + self._dims[:-1])).tolist()
        for i in np.arange(0, M):
            j = where(self._vars, vars_[i])
            exchange(self._vars, j, -(i+1))
            exchange(self._dims, j, -(i+1))
            exchange(c_r, j, -(i+1))   
        r_r = np.cumsum(([0] + self._dims[:-1])).tolist()
        
        P = block_permutation(r_r, c_r, self._dims)
        self._prec = (P)@(self._prec)@(P.T)
        self._info = (P)@(self._info)

    def _expand_scope(self, glob_vars, glob_dims):
        '''
        Expands the canonical forms scope to 
        accomodate new variables.

        Parameters
        ----------
        glob_vars: list
            The full set of variables the new scope
            must accomodate.
        glob_dims: list
            The respective dimensions of the global
            variables.
        Returns
        ----------
        K_prime: (d, d) ndarray
            A new precision matrix with expanded scope. 

        h_prime: (d, 1) ndarray
            A new information vector with expanded scope.
        '''
        A = np.zeros((sum(glob_dims), sum(self._dims)))

        columns = np.cumsum(([0] + self._dims[:-1])).tolist()
        rows = np.cumsum(([0] + glob_dims[:-1])).tolist()

        for i in np.arange(0, len(glob_vars)):
            if glob_vars[i] not in self._vars:
                rows.pop(i)

        for r, c, d in zip(rows, columns, self._dims):
            A[r:r+d, c:c+d] = np.identity(d)

        K_prime = (A)@(self._prec)@(A.T)
        h_prime = (A)@(self._info)

        return K_prime, h_prime

    def marginalize(self, vars_):
        '''
        Marginalize out the given set of variables.

        Parameters
        ----------
        vars_ : list
            The variables that are to summed out. Needs
            to be a subset of _vars.

        Returns
        ----------
        C : CanonicalGaussian
            A new potential with a reduced scope.
        '''
        self._rearrange(vars_)
        M = len(vars_)
        X = sum(self._dims[:-M])

        vars_ = self._vars[:-M]
        dims = self._dims[:-M]

        K_xx = self._prec[:X, :X]
        K_yy = np.linalg.inv(self._prec[X:, X:])
        K_xy = self._prec[:X, X:]
        K_yx = self._prec[X:, :X]

        h_x = self._info[:X]
        h_y = self._info[X:]

        K_prime = K_xx - (K_xy)@(K_yy)@(K_yx)
        h_prime = h_x - (K_xy)@(K_yy)@(h_y)
        g_prime = self._norm + 0.5*( np.log(np.linalg.det(2*np.pi*K_yy )) + (h_y.T)@(K_yy)@(h_y) )

        self._order()

        return CanonicalGaussian(vars_, dims, h_prime, K_prime, g_prime)

    def introduce_evidence(self, vars_, ev):
        '''
        Force a subset of variables into the given state.

        Parameters
        ----------
        vars_ : list
            The variables that are to summed out. Need 
            to be a subset of vars.
        ev : (d, 1) ndarray
            The state to which the given variables will
            be set.
        '''
        self._rearrange(vars_[::-1])
        M = len(vars_)
        X = sum(self._dims[:-M])
        N = sum(self._dims[M:])
        ev = ev.reshape((N, 1))

        self._vars = self._vars[:-M]
        self._dims = self._dims[:-M]

        K_xx = self._prec[:X, :X]
        K_yy = self._prec[X:, X:]
        K_xy = self._prec[:X, X:]

        h_x = self._info[:X]
        h_y = self._info[X:]

        self._prec = K_xx
        self._info = h_x - (K_xy)@(ev)
        self._norm += (h_y.T)@(ev) - 0.5*(ev.T)@(K_yy)@(ev)

        self._order()

    def __mul__(self, C):
        '''
        Overloads multiplication.

        Parameters
        ----------
        C : CanonicalGaussian or float
            The multiplicand.
        
        Returns
        ----------
        C : CanonicalGaussian
            The product of the two Gaussians.
        '''
        if isinstance(C, CanonicalGaussian):
            map_ = dict(zip(self._vars + C._vars, self._dims + C._dims))
            glob_vars, glob_dims = list(map_.keys()), list(map_.values())

            K_1, h_1 = self._expand_scope(glob_vars, glob_dims)
            K_2, h_2 = C._expand_scope(glob_vars, glob_dims)

            return CanonicalGaussian(glob_vars, glob_dims, h_1 + h_2, K_1 + K_2, self._norm + C._norm)
        else:
            return CanonicalGaussian(self._vars, self._dims, self._info, self._prec, self._norm + np.log(C))


    def __rmul__(self, C):
        '''
        Overloads reverse multiplication.

        Parameters
        ----------
        C : CanonicalGaussian or float
            The multiplicand.
        
        Returns
        ----------
        C : CanonicalGaussian
            The product of the two Gaussians.
        '''
        return self.__mul__(C)

    def __truediv__(self, C):
        '''
        Overloads division.

        Parameters
        ----------
        C : CanonicalGaussian
            The divisor.
        
        Returns
        ----------
        C : CanonicalGaussian
            The quotient of the two Gaussians.
        '''
        if isinstance(C, CanonicalGaussian):
            map_ = dict(zip(self._vars + C._vars, self._dims + C._dims))
            glob_vars = list(map_.keys())
            glob_dims = list(map_.values())

            K_1, h_1 = self._expand_scope(glob_vars, glob_dims)
            K_2, h_2 = C._expand_scope(glob_vars, glob_dims)

            return CanonicalGaussian(glob_vars, glob_dims, h_1 - h_2, K_1 - K_2, self._norm - C._norm)
        else:
            return CanonicalGaussian(self._vars, self._dims, self._info, self._prec, self._norm - np.log(C))

def block_permutation(rows, columns, dimensions):
    '''
    Creates a block permutation matrix for the given rows 
    and columns.

    Parameters
    ----------
    rows : list
        The beginning rows of each variables domain.
    cols : list
        The beginning columns of each variables domain.
    dimensions : list
        The respective list of the variables dimensions.
    '''
    N, M = len(dimensions), sum(dimensions)
    P = np.zeros((M, M))
    for r, c, d in zip(rows, columns, dimensions):
        P[r:r + d, c:c + d] = np.identity(d)
    return P

def where(list_, arg):
    '''
    A linear scan to find a variable's argument in
    a list. Python offers no better implementation.

    Parameters
    ----------
    list_ : list
        A list of variables.
    arg : int
        The variables values

    Returns
    ----------
    i : int
        The index of the first element equal to arg.
    '''
    for i in np.arange(0, len(list_)):
        if list_[i] == arg:
            return i

def exchange(list_, x, y):
    '''
    A helper function to exchange two positions
    in an array.

    Parameters
    ----------
    list_ : list
        A list.
    x : int
        The initial position.
    y : int
        The secondary position.
    '''
    tmp = list_[x]
    list_[x] = list_[y]
    list_[y] = tmp
