'''
A module containg the GaussianMixtureModel
class and some helper functions.

@author: scj robertson
@since: 29/05/16
'''

import numpy as np
from canonical_gaussian import CanonicalGaussian

class GaussianMixtureModel:
    '''
    Class for representing a mixture of canonical Gaussians.
    This is a fairly rough implementation.

    Parameters
    ----------
    mix : list
    	A list of CanonicalGaussians.

    Methods
    ----------
    marginalise
        Sums out a given subset of variables.
    introduce_evidence
    	Forces a subset of variables into the given state.

    Example
    ----------
    >>> print('To be completed, look at tracking.ipynb for now.')
    '''
    def __init__(self, mix):
        self._vars = []
        self._mix = list(mix)
        
        for gaussian in mix:
            self._vars += gaussian._vars
        self._vars = list(set(self._vars))

    def marginalise(self, vars_):
        print('!')

    def __mul__(self, gmm):
        '''
	Overloads multiplication.

	Parameters
	----------
        gmm : GaussianMixtureModel, CanonicalGaussian
	    The mutliplicands.
	Returns
	----------
	GMM : GaussianMixtureModels
	    The product of the mix of Gaussians.
        '''
        if isinstance(gmm, CanonicalGaussian): mixture = [gmm]
        else: mixture = gmm._mix
        product = [m*g for g in self._mix for m in mixture]
        return GaussianMixtureModel(product)

    def __rmul__(self, gmm):
        '''
	Overloads reverse multiplication.

	Parameters
	----------
        gmm : GaussianMixtureModel, CanonicalGaussian
	    The mutliplicands.
	Returns
	----------
	GMM : GaussianMixtureModels
	    The product of the mix of Gaussians.
        '''
        return self.__mul__(gmm)

    def __add__(self, gmm):
        '''
	Overloads addition.

	Parameters
	----------
	gmm : GaussianMixtureModel, CanonicalGaussian
	    The mutliplicands.
	Returns
	----------
	GMM : GaussianMixtureModels
	    The sum of the mix of Gaussians.
        '''
        if isinstance(gmm, CanonicalGaussian): mixture = [gmm]
        else: mixture = gmm._mix
        product = self._mix + mixture
        return GaussianMixtureModel(product)

    def __radd__(self, gmm):
        '''
	Overloads reverse addition.

	Parameters
	----------
	gmm : GaussianMixtureModel, CanonicalGaussian
	    The mutliplicands.
	Returns
	----------
	GMM : GaussianMixtureModels
	    The sum of the mix of Gaussians.
        '''
        return self.__add__(gmm)
