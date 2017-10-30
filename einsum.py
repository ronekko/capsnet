from collections import OrderedDict

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
    return in_subs, out_sub


def einsum(subscripts, *operands):
    """Computs the einsum for Chainer's Variable.
    Note that this is a very early implementation and supports only a few
    functionalities of original numpy.einsum function.
    """

    if not len(operands) == 2:
        raise ValueError(
            'Currently, `*operands` only supports the case of 2 arguments.')
    a, b = operands

    in_subs, out_subs = _parse_subscripts(subscripts)
    a_subs, b_subs = in_subs

    a_shape = OrderedDict()
    for sub, length in zip(a_subs, a.shape):
        a_shape[sub] = length
    b_shape = OrderedDict()
    for sub, length in zip(b_subs, b.shape):
        b_shape[sub] = length

    # Classify subscripts in a_subs and b_subs into 4 classes.
    # There are 4 sets of subscripts that are:
    # set_1: included in a_subs and b_subs and out_subs
    # set_2: included in a_subs and b_subs but not in out_subs
    # set_3a, set_3d: included in only a_subs or b_subs, and in out_subs
    # set_4a, set_4d: included in only a_subs or b_subs, and not in out_subs
    set_a = set(a_subs)
    set_b = set(b_subs)
    set_out = set(out_subs)

    set_1 = set_a & set_b
    set_3a = set_a - set_1
    set_3b = set_b - set_1
    set_2 = set_1 - set_out
    set_1 = set_1 - set_2
    set_4a = set_3a - set_out
    set_3a = set_3a - set_4a
    set_4b = set_3b - set_out
    set_3b = set_3b - set_4b

    #####################################################
    # Arrange a and b to prepare to input to F.batch_matmul
    # For the array a
    a_new_subs_class = ['', '', '']
    a_new_shape = [1, 1, 1]
    for sub, length in a_shape.items():
        if sub in set_1:
            a_new_subs_class[0] += sub
            a_new_shape[0] *= length
        elif sub in set_2:
            a_new_subs_class[2] += sub
            a_new_shape[2] *= length
        else:
            a_new_subs_class[1] += sub
            a_new_shape[1] *= length

    transpose_axes = []
    for sub in ''.join(a_new_subs_class):
        transpose_axes.append(a_subs.index(sub))

    a_ = F.transpose(a, transpose_axes)
    a_ = a_.reshape(a_new_shape)

    # For the array b
    b_new_subs_class = ['', '', '']
    b_new_shape = [1, 1, 1]
    for sub, length in b_shape.items():
        if sub in set_1:
            b_new_subs_class[0] += sub
            b_new_shape[0] *= length
        elif sub in set_2:
            b_new_subs_class[1] += sub
            b_new_shape[1] *= length
        else:
            b_new_subs_class[2] += sub
            b_new_shape[2] *= length

    transpose_axes = []
    for sub in ''.join(b_new_subs_class):
        transpose_axes.append(b_subs.index(sub))

    b_ = F.transpose(b, transpose_axes)
    b_ = b_.reshape(b_new_shape)

    #################
    # Target axes are reduced by F.batch_matmul and F.sum
    c_ = F.matmul(a_, b_)
    c_subs_class = [
        a_new_subs_class[0], a_new_subs_class[1], b_new_subs_class[2]]

    if set_4a:
        size_sum = 1
        for sub in set_4a:
            size_sum *= a_shape[sub]
            c_subs_class[1] = c_subs_class[1].replace(sub, '')
        size_kept = a_new_shape[1] // size_sum
        c_shape = c_.shape
        c_ = c_.reshape(c_shape[0], size_kept, size_sum, c_shape[2])
        c_ = F.sum(c_, axis=2)

    if set_4b:
        size_sum = 1
        for sub in set_4b:
            size_sum *= b_shape[sub]
            c_subs_class[2] = c_subs_class[2].replace(sub, '')
        size_kept = b_new_shape[2] // size_sum
        c_shape = c_.shape
        c_ = c_.reshape(c_shape[0], c_shape[1], size_kept, size_sum)
        c_ = F.sum(c_, axis=3)

    ############################
    # The output array is rearranged to output shape
    c_subs = ''.join(c_subs_class)
    sub_to_size = {**a_shape, **b_shape}  # merging two dicts
    c_shape = [sub_to_size[sub] for sub in c_subs]
    c = c_.reshape(c_shape)

    transpose_axes = []
    for sub in out_subs:
        transpose_axes.append(c_subs.index(sub))
    c = c.transpose(transpose_axes)
    return c
