import unittest

import numpy

import chainer
from chainer import cuda
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition

import einsum_function as einsum


@testing.parameterize(*testing.product_dict(
    [
        {'subs': 'ijk,ijk->ijk', 'operands_shapes': ((2, 3, 4), (2, 3, 4)),
         'y_shape': (2, 3, 4)},
        {'subs': 'ijk,ijk->ij', 'operands_shapes': ((2, 3, 4), (2, 3, 4)),
         'y_shape': (2, 3)},
        {'subs': 'bij,ij->bi', 'operands_shapes': ((5, 2, 4), (2, 4)),
         'y_shape': (5, 2)},
        {'subs': 'bij,ij->ij', 'operands_shapes': ((5, 2, 3), (2, 3)),
         'y_shape': (2, 3)},
        {'subs': 'bijk,ijk->', 'operands_shapes': ((5, 2, 3, 4), (2, 3, 4)),
         'y_shape': ()},
        {'subs': 'bijk,ijk', 'operands_shapes': ((5, 2, 3, 4), (2, 3, 4)),
         'y_shape': (5,)},
        {'subs': 'bij,ijk->bi', 'operands_shapes': ((5, 2, 3), (2, 3, 6)),
         'y_shape': (5, 2)},
        {'subs': 'bij,ijk->ij', 'operands_shapes': ((5, 2, 3), (2, 3, 6)),
         'y_shape': (2, 3)},
        {'subs': 'bij,ijk->bik', 'operands_shapes': ((5, 2, 3), (2, 3, 6)),
         'y_shape': (5, 2, 6)},
        {'subs': 'bi,ik->kb', 'operands_shapes': ((5, 2), (2, 6)),
         'y_shape': (6, 5)},
        {'subs': 'bij,ijk->jk', 'operands_shapes': ((5, 2, 3), (2, 3, 6)),
         'y_shape': (3, 6)},
        {'subs': 'ij,ij->ij', 'operands_shapes': ((2, 3), (2, 3)),
         'y_shape': (2, 3)},
        {'subs': ' ij , ij -> ij ', 'operands_shapes': ((2, 3), (2, 3)),
         'y_shape': (2, 3)},
    ],
    [
#        {'dtype': numpy.float16},  # TODO: fix np.einsum for float16
        {'dtype': numpy.float32},
        {'dtype': numpy.float64},
    ],
))
class TestEinsum(unittest.TestCase):
    def setUp(self):
        self.operands = []
        for shape in self.operands_shapes:
            operand = numpy.random.uniform(-1, 1, shape).astype(self.dtype)
            self.operands.append(operand)
        self.gy = numpy.random.uniform(-1, 1, self.y_shape).astype(self.dtype)

        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 2e-3}
            self.check_backward_options = {'atol': 2e-3}
        else:
            self.check_forward_options = {}
            self.check_backward_options = {}

    def check_forward(self, subscripts, operands_data):
        xp = cuda.get_array_module(*operands_data)
        operands = [
            chainer.Variable(operand_data) for operand_data in operands_data]
        y = einsum.einsum(subscripts, *operands)
        self.assertEqual(y.data.dtype, self.dtype)
        y_expect = xp.asarray(numpy.einsum(self.subs, *self.operands))
        testing.assert_allclose(y_expect, y.data,
                                **self.check_forward_options)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.subs, self.operands)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        operands = [cuda.to_gpu(operand) for operand in self.operands]
        self.check_forward(self.subs, operands)

    def check_backward(self,  subscripts, operands_data, y_grad):
        print(subscripts)
        gradient_check.check_backward(
            lambda *operands_: einsum.einsum(subscripts, *operands_),
            operands_data, y_grad,
            dtype=numpy.float64, **self.check_backward_options)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.subs, self.operands, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        operands = [cuda.to_gpu(operand) for operand in self.operands]
        self.check_backward(self.subs, operands, cuda.to_gpu(self.gy))


@testing.parameterize(*testing.product_dict(
    [
        {'sub': 'ijk,ijk', 'in_subs': ['ijk', 'ijk'], 'out_sub': '',
         'broadcast_subs': ['', '']},
        {'sub': 'ijk,ijk->ij', 'in_subs': ['ijk', 'ijk'], 'out_sub': 'ij',
         'broadcast_subs': ['', '']},
        {'sub': 'bijk,ijk->bi', 'in_subs': ['bijk', 'ijk'], 'out_sub': 'bi',
         'broadcast_subs': ['', '']},
        {'sub': 'bij,ij->', 'in_subs': ['bij', 'ij'], 'out_sub': '',
         'broadcast_subs': ['b', '']},
        {'sub': 'bij,ij', 'in_subs': ['bij', 'ij'], 'out_sub': 'b',
         'broadcast_subs': ['', '']},
        {'sub': ' bijk , ijk -> bi ',
         'in_subs': ['bijk', 'ijk'], 'out_sub': 'bi',
         'broadcast_subs': ['', '']},
    ]
))
class TestParseSubscripts(unittest.TestCase):

    def check_parse_subscripts(
            self, subscripts, expected_in_subs, expected_out_sub,
            expected_broadcast_subs):
        in_subs, out_sub, broadcast_subs = einsum._parse_subscripts(subscripts)
        self.assertEqual(out_sub, expected_out_sub)
        for in_sub, expected_in_sub in zip(in_subs, expected_in_subs):
            assert in_sub == expected_in_sub, '''
    actual  : {}
    expected: {}'''.format(in_sub, expected_in_sub)
        assert out_sub == expected_out_sub, '''
    actual  : {}
    expected: {}'''.format(out_sub, expected_out_sub)
        for broadcast_sub, expected_broadcast_sub in zip(
                broadcast_subs, expected_broadcast_subs):
            assert broadcast_sub == expected_broadcast_sub, '''
    actual  : {}
    expected: {}'''.format(broadcast_sub, expected_broadcast_sub)

    def test_parse_subscripts(self):
        self.check_parse_subscripts(self.sub, self.in_subs, self.out_sub,
                                    self.broadcast_subs)


testing.run_module(__name__, __file__)
