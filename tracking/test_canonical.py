'''
Test the CanonicalGaussian class.

@author: scj
@since: 25/05/2016
'''

from nose.tools import assert_equals
from canonical_gaussian import CanonicalGaussian
import numpy as np
import numpy.testing as npt

class TestCanonicalGaussian:

    def setup(self):
        print('TestCanonicalGaussian.setup()')
        self.K = np.array([[1, -1, 0], [-1, 4, -2], [0, -2, 4]])
        self.h = np.array([[1, 4, -1]])
        self.g = -2
        self.vars_ = [3, 1, 2]
        self.dims =[1, 1, 1]
        self.gauss = CanonicalGaussian(self.vars_, self.dims, self.h, self.K, self.g)
        self.ev = np.array([[2], [1]])

        self.K_1 = np.array([[1, -1], [-1, 1]])
        self.h_1 = np.array([[1], [-1]])
        self.g_1 = -3
        self.vars_1 = [1, 2]
        self.dims_1 = [1, 1]
        self.C_1 = CanonicalGaussian(self.vars_1, self.dims_1, self.h_1, self.K_1, self.g_1)
	
        self.K_2 = np.array([[3, -2], [-2, 4]])
        self.h_2 = np.array([[5], [-1]])
        self.g_2 = 1
        self.vars_2 = [2, 3]
        self.dims_2 = [1, 1]
        self.C_2 = CanonicalGaussian(self.vars_2, self.dims_2, self.h_2, self.K_2, self.g_2)
	

    def teardown(self):
        print('TestCanonicalGaussian.teardown()')

    @classmethod
    def setup_class(cls):
        print('setup_class()')

    @classmethod
    def teardown_class(cls):
        print('teardown_class()')

    def test_prec(self):
        print('TestCanonicalGaussian.test_prec()')
        K_prime = np.array([[4, -2, -1], [-2, 4, 0], [-1, 0, 1]])
        npt.assert_array_equal(self.gauss._prec, K_prime)

    def test_info(self):
        print('TestCanonicalGaussian.test_info()')
        h_prime = np.array([[4], [-1], [1]])
        npt.assert_array_equal(self.gauss._info, h_prime)

    def test_rearrange(self):
        print('TestCanonicalGaussian.test_rearrange()')
        K_prime = np.array([[4, -1, -2], [-1, 1, 0], [-2, 0, 4]])
        self.gauss._rearrange([3, 2])
        npt.assert_array_equal(self.gauss._prec, K_prime)

    def test_marginalize_prec(self):
        print('TestCanonicalGaussian.test_marginalize_prec()')
        K_prime = np.array([[2]])
        C = self.gauss.marginilize([2, 3])
        npt.assert_array_equal(C._prec, K_prime)

    def test_marginalize_info(self):
        print('TestCanonicalGaussian.test_marginalize_prec()')
        h_prime = np.array([[4.5]])
        C = self.gauss.marginilize([2, 3])
        npt.assert_array_equal(C._info, h_prime)

    def test_evidence_prec(self):
        print('TestCanonicalGaussian.test_evidence_prec()')
        K_prime = np.array([[4]])
        self.gauss.introduce_evidence([2, 3], self.ev)
        npt.assert_array_equal(self.gauss._prec, K_prime)

    def test_evidence_info(self):
        print('TestCanonicalGaussian.test_evidence_info()')
        h_prime = np.array([[9]])
        self.gauss.introduce_evidence([2, 3], self.ev)
        npt.assert_array_equal(self.gauss._info, h_prime)

    def test_multiplication(self):
        print('TestCanonicalGaussian.test_multiplication()')
        C = self.C_1*self.C_2
