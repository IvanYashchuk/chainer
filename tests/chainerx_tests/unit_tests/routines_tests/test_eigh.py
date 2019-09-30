import numpy
import pytest

import chainer
import chainerx
import chainerx.testing

from chainerx_tests import array_utils
from chainerx_tests import dtype_utils
from chainerx_tests import op_utils


class NumpyLinalgOpTest(op_utils.NumpyOpTest):

    dodge_nondifferentiable = True

    def setup(self):
        device = chainerx.get_default_device()
        if (device.backend.name == 'native'
                and not chainerx.linalg._is_lapack_available()):
            pytest.skip('LAPACK is not linked to ChainerX')
        self.check_backward_options.update({
            'eps': 1e-5, 'rtol': 1e-3, 'atol': 1e-3})
        self.check_double_backward_options.update({
            'eps': 1e-5, 'rtol': 1e-3, 'atol': 1e-3})


_numpy_does_not_support_0d_input113 = \
    numpy.lib.NumpyVersion(numpy.__version__) < '1.13.0'


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    chainer.testing.product({
        'shape': [(0, 0), (1, 1), (3, 3), (6, 6)],
        'in_dtypes': ['float32', 'float64'],
        'UPLO': ['u', 'L']
    })
))
class TestEigh(NumpyLinalgOpTest):

    def generate_inputs(self):
        singular_values = numpy.random.uniform(
            low=0.1, high=1.5, size=self.shape[0])
        a = chainer.testing.generate_matrix(
            self.shape, self.in_dtypes, singular_values=singular_values)
        return a,

    def forward_xp(self, inputs, xp):
        a, = inputs

        if (_numpy_does_not_support_0d_input113 and a.size == 0):
            pytest.skip('Older NumPy versions do not work with empty arrays')

        # Input has to be symmetrized for backward test to work
        def symmetrize(A):
            L = xp.tril(A)
            return (L + L.T)/2.
        a = symmetrize(a)

        w, v = xp.linalg.eigh(a, UPLO=self.UPLO)

        # The sign of eigenvectors is not unique,
        # therefore absolute values are compared
        return w, xp.abs(v)
