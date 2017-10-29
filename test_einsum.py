import unittest

import numpy

import chainer
from chainer import cuda
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition

import einsum


@testing.parameterize(*testing.product_dict(
    [
        {'subs': 'ijk,ijk->ijk', 'a_shape': (2, 3, 4), 'b_shape': (2, 3, 4),
         'y_shape': (2, 3, 4)},
        {'subs': 'ijk,ijk->ij', 'a_shape': (2, 3, 4), 'b_shape': (2, 3, 4),
         'y_shape': (2, 3)},
        {'subs': 'bij,ij->bi', 'a_shape': (5, 2, 4), 'b_shape': (2, 4),
         'y_shape': (5, 2)},
        {'subs': 'bij,ij->ij', 'a_shape': (5, 2, 3), 'b_shape': (2, 3),
         'y_shape': (2, 3)},
        {'subs': 'bijk,ijk->', 'a_shape': (5, 2, 3, 4), 'b_shape': (2, 3, 4),
         'y_shape': ()},
        {'subs': 'bijk,ijk', 'a_shape': (5, 2, 3, 4), 'b_shape': (2, 3, 4),
         'y_shape': (5,)},
        {'subs': 'bij,ijk->bi', 'a_shape': (5, 2, 3), 'b_shape': (2, 3, 6),
         'y_shape': (5, 2)},
        {'subs': 'bij,ijk->ij', 'a_shape': (5, 2, 3), 'b_shape': (2, 3, 6),
         'y_shape': (2, 3)},
        {'subs': 'bij,ijk->bik', 'a_shape': (5, 2, 3), 'b_shape': (2, 3, 6),
         'y_shape': (5, 2, 6)},
        {'subs': 'bi,ik->kb', 'a_shape': (5, 2), 'b_shape': (2, 6),
         'y_shape': (6, 5)},
        {'subs': 'bij,ijl->jl', 'a_shape': (5, 2, 3), 'b_shape': (2, 3, 6),
         'y_shape': (3, 6)},
        {'subs': 'ij,ij->ij', 'a_shape': (2, 3), 'b_shape': (2, 3),
         'y_shape': (2, 3)},
        {'subs': ' ij , ij -> ij ', 'a_shape': (2, 3), 'b_shape': (2, 3),
         'y_shape': (2, 3)},
    ],
    [
#        {'dtype': numpy.float16},
        {'dtype': numpy.float32},
        {'dtype': numpy.float64},
    ],
))
class TestEinsum(unittest.TestCase):
    def setUp(self):
        self.a = numpy.random.uniform(-1, 1, self.a_shape).astype(self.dtype)
        self.b = numpy.random.uniform(-1, 1, self.b_shape).astype(self.dtype)
        self.gy = numpy.random.uniform(-1, 1, self.y_shape).astype(self.dtype)

        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 2e-3}
            self.check_backward_options = {'atol': 2e-3}
        else:
            self.check_forward_options = {}
            self.check_backward_options = {}

    def check_forward(self, subscripts, a_data, b_data):
        xp = cuda.get_array_module(a_data)
        a = chainer.Variable(a_data)
        b = chainer.Variable(b_data)
        y = einsum.einsum(subscripts, a, b)
        self.assertEqual(y.data.dtype, self.dtype)
        y_expect = xp.asarray(numpy.einsum(self.subs, self.a, self.b))
        testing.assert_allclose(y_expect, y.data,
                                **self.check_forward_options)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.subs, self.a, self.b)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(self.subs, cuda.to_gpu(self.a), cuda.to_gpu(self.b))

    def check_backward(self,  subscripts, a_data, b_data, y_grad):
        gradient_check.check_backward(
            lambda a_, b_: einsum.einsum(subscripts, a_, b_),
            (a_data, b_data), y_grad,
            dtype=numpy.float64, **self.check_backward_options)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.subs, self.a, self.b, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(self.subs, cuda.to_gpu(self.a),
                            cuda.to_gpu(self.b), cuda.to_gpu(self.gy))


@testing.parameterize(*testing.product_dict(
    [
        {'sub': 'ijk,ijk', 'in_subs': ['ijk', 'ijk'], 'out_sub': ''},
        {'sub': 'ijk,ijk->ij', 'in_subs': ['ijk', 'ijk'], 'out_sub': 'ij'},
        {'sub': 'bijk,ijk->bi', 'in_subs': ['bijk', 'ijk'], 'out_sub': 'bi'},
        {'sub': 'bij,ij->', 'in_subs': ['bij', 'ij'], 'out_sub': ''},
        {'sub': 'bij,ij', 'in_subs': ['bij', 'ij'], 'out_sub': 'b'},
        {'sub': ' bijk , ijk -> bi ',
         'in_subs': ['bijk', 'ijk'], 'out_sub': 'bi'},
    ]
))
class TestParseSubscripts(unittest.TestCase):

    def check_parse_subscripts(
            self, subscripts, expected_in_subs, expected_out_sub):
        in_subs, out_sub = einsum._parse_subscripts(subscripts)
        self.assertEqual(out_sub, expected_out_sub)
        for in_sub, expected_in_sub in zip(in_subs, expected_in_subs):
            assert in_sub == expected_in_sub, '''
    actual  : {}
    expected: {}'''.format(in_sub, expected_in_sub)
        assert out_sub == expected_out_sub, '''
    actual  : {}
    expected: {}'''.format(out_sub, expected_out_sub)

    def test_parse_subscripts(self):
        self.check_parse_subscripts(self.sub, self.in_subs, self.out_sub)


testing.run_module(__name__, __file__)
