from collections import OrderedDict

import numpy
import chainer
import chainer.functions as F


def _parse_subscripts(subscripts):
    if '->' in subscripts:
        has_arrow = True
    else:
        subscripts = subscripts + '->'
        has_arrow = False

    in_subs, out_sub = subscripts.split('->')
    in_subs = in_subs.split(',')
    out_sub = out_sub.strip()
    in_subs = [in_sub.strip() for in_sub in in_subs]

    if not has_arrow:
        out_sub = ''
        shared_chars = set.intersection(*[set(sub) for sub in in_subs])
        for subs in in_subs:
            for char in subs:
                if char not in shared_chars:
                    out_sub += char

    # 片方のみに含まれる添字(すなわち単純sumの添字)をout_subに付け足す
    broadcast_subs = []
    in_sub_sets = [set(sub) for sub in in_subs]
    for i, sub_set in enumerate(in_sub_sets):
        others = in_sub_sets.copy()
        others.pop(i)
        others = set.union(set(out_sub), *others)
        broadcast_sub_set = sub_set - others
        broadcast_subs.append(''.join(broadcast_sub_set))
    return in_subs, out_sub, broadcast_subs


class Einsum(chainer.Function):
    def __init__(self, subscripts):
        self.subscripts = subscripts

    def forward(self, inputs):
        self.retain_inputs(tuple(range(len(inputs))))
        xp = chainer.cuda.get_array_module(*inputs)
        y = xp.einsum(self.subscripts, *inputs)
        if xp is numpy:
            y = numpy.asarray(y)
        return y,

    def backward(self, inputs, grad_outputs):
        xp = chainer.cuda.get_array_module(*inputs)
        a, b = inputs
        gy, = grad_outputs

        broadcast_sub_sizes = []
        in_subs, out_sub, broadcast_subs = _parse_subscripts(self.subscripts)
        for in_array, in_sub, bc_sub in zip(inputs, in_subs, broadcast_subs):
            sub_to_size = {}
            for sub in bc_sub:
                axis = in_sub.find(sub)
                if axis != -1:
                    sub_to_size[sub] = in_array.shape[axis]
            broadcast_sub_sizes.append(sub_to_size)


        # TODO: Make below code clean (and simple)
        a_sub, b_sub = in_subs
        # for ga
        sub_to_size = broadcast_sub_sizes[0]
        gy_tmp = gy.reshape(gy.shape + (1,) * len(sub_to_size))
        gy_tmp = xp.broadcast_to(gy_tmp, gy.shape + tuple(sub_to_size.values()))
        out_sub_tmp = out_sub + ''.join(sub_to_size.keys())
        ga = xp.einsum('{},{}->{}'.format(out_sub_tmp, b_sub, a_sub), gy_tmp, b)
        # for gb
        sub_to_size = broadcast_sub_sizes[1]
        gy_tmp = gy.reshape(gy.shape + (1,) * len(sub_to_size))
        gy_tmp = xp.broadcast_to(gy_tmp, gy.shape + tuple(sub_to_size.values()))
        out_sub_tmp = out_sub + ''.join(sub_to_size.keys())
        gb = xp.einsum('{},{}->{}'.format(out_sub_tmp,a_sub,  b_sub), gy_tmp, a)

        return ga, gb


def einsum(subscripts, *operands):
    return Einsum(subscripts)(*operands)


if __name__ == '__main__':
    import numpy as np

    xp = np
#    subs = 'bij,ijk->jk'
#    operands_shapes = ((5, 2, 3), (2, 3, 6))
    subs = 'bijk,ijk->'
    operands_shapes = ((5, 2, 3, 4), (2, 3, 4))

    operands = []
    for shape in operands_shapes:
        operands.append(chainer.Variable(np.random.randn(*shape).astype('f')))

    y = einsum(subs, *operands)
    y.grad = np.ones_like(y.array)
    y.backward()
