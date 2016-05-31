'''
A module containing the GaussianMixtureModel
class and some helper functions.

@author: scj robertson
@since: 29/05/16
'''
import numpy as np
from canonical_gaussian import CanonicalGaussian

class GaussianMixtureModel:
    '''
    Class for representing a mixture of canonical Gaussians.
    This is rough implementation.

    Parameters
    ----------
    mix : list
    	A list of CanonicalGaussians.

    Methods
    ----------
    marginalize
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
        self._N = len(mix)
        
        for gaussian in mix:
            self._vars += gaussian._vars
        self._vars = list(set(self._vars))

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
        result = [g.marginalize(vars_) for g in self._mix if set(vars_).issubset(set(g._vars))]
        return GaussianMixtureModel(result)

    def __mul__(self, gmm):
        '''
	Overloads multiplication.

	Parameters
	----------
        gmm : GaussianMixtureModel, CanonicalGaussian
	    The mutliplicands.
	Returns
	----------
	GMM : GaussianMixtureModel
	    The product of the mix of Gaussians.
        '''
        mixture = []
        if isinstance(gmm, CanonicalGaussian): mixture = [gmm]
        elif isinstance(gmm, GaussianMixtureModel): mixture = gmm._mix
        product = [g*m for g in self._mix for m in mixture]
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
	GMM : GaussianMixtureModel
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
	GMM : GaussianMixtureModel
	    The sum of the mix of Gaussians.
        '''
        if isinstance(gmm, CanonicalGaussian): mixture = [gmm]
        else: mixture = gmm._mix
        sum_ = self._mix + mixture
        return GaussianMixtureModel(sum_)

    def __radd__(self, gmm):
        '''
	Overloads reverse addition.

	Parameters
	----------
	gmm : GaussianMixtureModel, CanonicalGaussian
	    The mutliplicands.
	Returns
	----------
	GMM : GaussianMixtureModel
	    The sum of the mix of Gaussians.
        '''
        return self.__add__(gmm)
