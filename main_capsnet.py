# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 12:11:21 2017

@author: ryuhei

A Chainer implementation of
"Dynamic Routing Between Capsules"
"""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, Variable
from chainer import initializers
from chainer import optimizers
from datasets import load_mnist_as_ndarray, random_augment_padding


class CapsNet(chainer.Chain):
    def __init__(self, routing_iterations=3):
        super(CapsNet, self).__init__()

        with self.init_scope():
            self.conv1 = L.Convolution2D(1, 256, 9)
            self.primary_caps = PrimaryCapsules()
            self.digit_caps = DigitCaps(routing_iterations)
            self.decoder = Decoder()

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        u = self.primary_caps(h)
        v = self.digit_caps(u)
        return v


class PrimaryCapsules(chainer.Chain):
    def __init__(self, initialW=None):
        self.U = 8   # number of dimensions of a capsule
        self.C = 32  # number of channels
        self.H, self.W = 6, 6  # height and width of the output feature map

        super(PrimaryCapsules, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(256, self.C * self.U, 9, stride=2)

    def __call__(self, x):
        B = len(x)
        C, U, H, W = self.C, self.U, self.H, self.W

        h = self.conv(x)  # (B, CU, H, W)
        h = h.reshape(B, C, U, H, W)
        u = F.transpose(h, (0, 1, 3, 4, 2))
        return u


class DigitCaps(chainer.Chain):
    def __init__(self, routing_iterations=3,
                 initial_W=None, learn_b_init=False, initial_b_init=None):
        U = 8   # number of dimensions of input capsule
        V = 16   # number of dimensions of output capsule

        C = 32  # number of channels
        H, W = 6, 6
        I = C * H * W  # num. of input capsules (i.e. CHW size of feature map)
        J = 10  # number of output capsules (i.e. classes)

        super(DigitCaps, self).__init__()
        with self.init_scope():
            W_initializer = initializers._get_initializer(initial_W)
            self.W = chainer.Parameter(W_initializer, (I, J * V, U), 'W')
            if learn_b_init:
                if initial_b_init is None:
                    initial_b_init = 0
                b_initializer = initializers._get_initializer(initial_b_init)
                self.b_init = chainer.Parameter(
                    b_initializer, (J, I), 'b_init')

        self.routing_iterations = routing_iterations
        self.learn_b_init = learn_b_init
        self.V, self.J = V, J

    def __call__(self, u):
        V, J = self.V, self.J
        B, C, H, W, U = u.shape
        I = C * H * W

        # Change u.shape from (B, C, H, W, U) -> (CHW, B, U) = (I, B, U)
        u = u.reshape(B, C * H * W, U)
        u = F.transpose(u, (1, 0, 2))

        u_hat = F.batch_matmul(u, self.W, transb=True)
        # Change u_hat.shape from (I, B, JV) -> (BJ, I, V)
        u_hat = u_hat.reshape(I, B, J, V)
        u_hat = F.transpose(u_hat, (1, 2, 0, 3))
        u_hat = u_hat.reshape(B * J, I, V)

        # Procedure 1
        xp = self.xp
        if self.learn_b_init:
            b = self.b_init
        else:
            b = chainer.Variable(xp.zeros((J, I), dtype='f'))
        b = F.tile(b, (B, 1, 1))  # (B, J, I)
        b = b.reshape(B * J, I)  # (BJ, I)

        for r in range(self.routing_iterations):
            c = F.softmax(b, axis=1)  # (BJ, I)
            # (BJ, I) @ (BJ, I, V) -> (BJ, V)
            s = F.batch_matmul(c, u_hat, transa=True).reshape(B * J, V)
            s = s.reshape(B * J, V)
            v = squash(s)  # (BJ, V)
            if r < (self.routing_iterations - 1):  # skip at the last iteration
                b += F.batch_matmul(u_hat, v).reshape(B * J, I)  # (BJ, I)

        v = v.reshape(B, J, V)
        return v


class Decoder(chainer.Chain):
    def __init__(self):

        super(Decoder, self).__init__()
        with self.init_scope():
            self.fc_1 = L.Linear(16, 512)
            self.fc_2 = L.Linear(512, 1024)
            self.fc_3 = L.Linear(1024, 784)

    def __call__(self, x):
        h = F.relu(self.fc_1(x))
        h = F.relu(self.fc_2(h))
        return F.sigmoid(self.fc_3(h))


def squash(x):
    sq_norm = F.batch_l2_norm_squared(x)
    factor = F.sqrt(sq_norm) / (1 + sq_norm)
    return F.broadcast_to(F.expand_dims(factor, 1), x.shape) * x


def batch_l2_norm(x):
    return F.sqrt(F.batch_l2_norm_squared(x))


def separate_margin_loss(x, t, m_posi=0.9, m_nega=0.1, p_lambda=0.5):
    batch_size, num_classes, dim_capsules = x.shape
    xp = cuda.get_array_module(x)
    mask = xp.zeros((batch_size, num_classes), dtype=np.bool)
    mask[list(range(batch_size)), t] = True  # one-hot mask

    x = x.reshape(-1, dim_capsules)
    mask = mask.ravel()
    loss_posi = F.relu(m_posi - batch_l2_norm(x[mask])) ** 2
    loss_nega = F.relu(batch_l2_norm(x[~mask]) - m_nega) ** 2
    loss = F.sum(loss_posi) + p_lambda * F.sum(loss_nega)
    return loss


def accuracy(x, t):
    batch_size, num_classes, dim_capsules = x.shape
    x = x.reshape(-1, dim_capsules)
    x_norms = F.batch_l2_norm_squared(x).reshape(batch_size, num_classes)
    pred = x_norms.data.argmax(axis=1)
    return Variable((pred == t).mean())


if __name__ == '__main__':
    # Hyperparameters
    p = SimpleNamespace()
    p.gpu = 0
    p.num_epochs = 300
    p.batch_size = 80
    p.lr = 0.001
    p.weight_decay = 0  # 1e-4
    p.da_pad = 2
    p.da_mirror = False
    p.routing_iterations = 3
    p.weight_loss_recon = 5e-4

    xp = np if p.gpu < 0 else chainer.cuda.cupy

    # Dataset
    train, test = load_mnist_as_ndarray(3)
    x_train, c_train = train
    x_test, c_test = test
    num_train = len(x_train)
    num_test = len(x_test)
#    std_rgb = x_train.std((0, 2, 3), keepdims=True)
#    x_train /= std_rgb
#    x_test /= std_rgb
#    mean_rgb = x_train.mean((0, 2, 3), keepdims=True)
#    x_train -= mean_rgb
#    x_test -= mean_rgb

    # Model and optimizer
    model = CapsNet()
    if p.gpu >= 0:
        model.to_gpu()
    optimizer = optimizers.Adam(p.lr)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(p.weight_decay))

    # Training loop
    train_loss_log = []
    train_acc_log = []
    test_loss_log = []
    test_acc_log = []
    best_test_acc = 0
    try:
        for epoch in range(p.num_epochs):
            epoch_losses = []
            epoch_accs = []
            for i in tqdm(range(0, num_train, p.batch_size)):
                x_batch = random_augment_padding(x_train[i:i+p.batch_size],
                                                 p.da_pad, p.da_mirror)
                x_batch = xp.asarray(x_batch)
                c_batch = xp.asarray(c_train[i:i+p.batch_size])
                model.cleargrads()
                with chainer.using_config('train', True):
                    y_batch = model(x_batch)
                    loss_margin = separate_margin_loss(y_batch, c_batch)
                    y_c = y_batch[list(range(len(y_batch))), c_batch]
                    x_recon = model.decoder(y_c)
                    loss_recon = F.mean_squared_error(x_recon, x_batch)
                    loss = loss_margin + p.weight_loss_recon * loss_recon
                    acc = accuracy(y_batch, c_batch)
                    loss.backward()
                optimizer.update()
                epoch_losses.append(loss.data)
                epoch_accs.append(acc.data)

            epoch_loss = np.mean(chainer.cuda.to_cpu(xp.stack(epoch_losses)))
            epoch_acc = np.mean(chainer.cuda.to_cpu(xp.stack(epoch_accs)))
            train_loss_log.append(epoch_loss)
            train_acc_log.append(epoch_acc)

            # Evaluate the test set
            losses = []
            accs = []
            for i in tqdm(range(0, num_test, p.batch_size)):
                x_batch = xp.asarray(x_test[i:i+p.batch_size])
                c_batch = xp.asarray(c_test[i:i+p.batch_size])
                with chainer.no_backprop_mode():
                    with chainer.using_config('train', False):
                        y_batch = model(x_batch)
                        loss = separate_margin_loss(y_batch, c_batch)
                        acc = accuracy(y_batch, c_batch)
                losses.append(loss.data)
                accs.append(acc.data)
            test_loss = np.mean(chainer.cuda.to_cpu(xp.stack(losses)))
            test_acc = np.mean(chainer.cuda.to_cpu(xp.stack(accs)))
            test_loss_log.append(test_loss)
            test_acc_log.append(test_acc)

            # Keep the best model so far
            if test_acc > best_test_acc:
                best_model = deepcopy(model)
                best_test_loss = test_loss
                best_test_acc = test_acc
                best_epoch = epoch

            # Display the training log
            print('{}: loss = {}'.format(epoch, epoch_loss))
            print('test acc = {}'.format(test_acc))
            print('best test acc = {} (# {})'.format(best_test_acc,
                                                     best_epoch))
            print(p)

            plt.figure(figsize=(10, 4))
            plt.title('Loss')
            plt.plot(train_loss_log, label='train loss')
            plt.plot(test_loss_log, label='test loss')
            plt.legend()
            plt.grid()
            plt.show()

            plt.figure(figsize=(10, 4))
            plt.title('Accucary')
            plt.plot(train_acc_log, label='train acc')
            plt.plot(test_acc_log, label='test acc')
            plt.legend()
            plt.grid()
            plt.show()

    except KeyboardInterrupt:
        print('Interrupted by Ctrl+c!')

    print('best test acc = {} (# {})'.format(best_test_acc,
                                             best_epoch))
    print(p)
    print(optimizer)
