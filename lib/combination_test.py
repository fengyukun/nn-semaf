#! /usr/bin/env python3
"""
Authors: fengyukun
Date:   2016-06-02
Brief:  Test the combination of layers
"""

# Activate automatic float divison for python2.
from __future__ import division
from inc import*
from gradient_checker import GradientChecker
import layer
from layer import HiddenLayer
from layer import SoftmaxLayer
from lstm_layer import LSTMLayer
from recurrent_layer import RecurrentLayer


class FeedForwardNNTest(object):
    """
    FeedForwardNNTest class
    """
    def __init__(self, n_i, no_hidden, no_softmax, use_bias, act_func):
        """
        Init nn test
        """
        self.n_i = n_i
        self.no_hidden = no_hidden
        self.no_softmax = no_softmax
        self.use_bias = use_bias
        self.act_func = act_func
        self.hidden_layer = HiddenLayer()
        self.hidden_layer.init_layer(n_i=self.n_i, n_o=self.no_hidden,
                                     act_func=self.act_func,
                                     use_bias=self.use_bias)
        self.softmax_layer = SoftmaxLayer()
        self.softmax_layer.init_layer(n_i=self.no_hidden, n_o=self.no_softmax,
                                      use_bias=use_bias)
        self.params = self.hidden_layer.params + self.softmax_layer.params
        self.param_names = (self.hidden_layer.param_names +
                            self.softmax_layer.param_names)

    def cost(self, x, y):
        """
        Cost function
        """
        py = self.forward(x)
        cross_entropy = -np.sum(
            np.log(py[np.arange(0, y.shape[0]), y])
        )
        return cross_entropy

    def forward(self, x):
        """
        Compute forward pass
        """

        hidden_out = self.hidden_layer.forward(x)
        softmax_out = self.softmax_layer.forward(hidden_out)

        # Keep track output
        self.softmax_out = softmax_out
        return softmax_out

    def backprop(self, y):
        """
        Back propagation. Computing gradients on parameters of nn
        y: numpy.ndarray
            Normalized correct label of x
        """

        if not hasattr(self, 'softmax_out'):
            logging.error("No forward pass is computed")
            raise Exception

        go = np.zeros(self.softmax_out.shape)
        for i in range(0, go.shape[0]):
            go[i][y[i]] = (-1) / self.softmax_out[i][y[i]]

        ghidden = self.softmax_layer.backprop(go)
        gx = self.hidden_layer.backprop(ghidden)

        self.gparams = self.hidden_layer.gparams + self.softmax_layer.gparams

        return gx


class RecurrentNNTest(object):
    """
    RecurrentNNTest class
    """
    def __init__(self, n_i, no_hidden, no_softmax, use_bias, act_func):
        """
        Init nn test
        """
        self.n_i = n_i
        self.no_hidden = no_hidden
        self.no_softmax = no_softmax
        self.use_bias = use_bias
        self.act_func = act_func

        self.recurrent_layer = LSTMLayer()
        # self.recurrent_layer = RecurrentLayer()
        self.recurrent_layer.init_layer(n_i=self.n_i, n_o=self.no_hidden,
                                        act_func=self.act_func,
                                        use_bias=use_bias)

        self.softmax_layer = SoftmaxLayer()
        self.softmax_layer.init_layer(n_i=self.no_hidden, n_o=self.no_softmax,
                                      use_bias=use_bias)
        self.params = self.recurrent_layer.params + self.softmax_layer.params
        self.param_names = (self.recurrent_layer.param_names +
                            self.softmax_layer.param_names)

    def cost(self, x, y):
        """
        Cost function
        """
        py = self.forward(x)
        cross_entropy = -np.sum(
            np.log(py[np.arange(0, y.shape[0]), y])
        )
        return cross_entropy

    def forward(self, x):
        """
        Compute forward pass
        """

        hidden_out = self.recurrent_layer.forward(
            x, starts=None, ends=None, reverse=False, output_opt='last'
        )

        softmax_out = self.softmax_layer.forward(hidden_out)

        # Keep track output
        self.softmax_out = softmax_out
        return softmax_out

    def backprop(self, y):
        """
        Back propagation. Computing gradients on parameters of nn
        y: numpy.ndarray
            Normalized correct label of x
        """

        if not hasattr(self, 'softmax_out'):
            logging.error("No forward pass is computed")
            raise Exception

        go = np.zeros(self.softmax_out.shape)
        for i in range(0, go.shape[0]):
            go[i][y[i]] = (-1) / self.softmax_out[i][y[i]]

        ghidden = self.softmax_layer.backprop(go)
        gx = self.recurrent_layer.backprop(ghidden)

        self.gparams = (self.recurrent_layer.gparams +
                        self.softmax_layer.gparams)

        return gx


def feedforward_nntest():
    n_i = 5
    no_hidden = 10
    no_softmax = 5
    use_bias = True
    act_func = 'tanh'
    x_num = 10
    x = np.random.uniform(low=0, high=5, size=(x_num, n_i))
    y = np.random.randint(low=0, high=no_softmax, size=x_num)
    nntest = FeedForwardNNTest(n_i, no_hidden, no_softmax, use_bias, act_func)

    gc = GradientChecker()
    gc.check_nn(nntest, x, y)


def recurrent_nntest():
    n_i = 5
    no_hidden = 10
    no_softmax = 5
    use_bias = True
    act_func = 'tanh'
    x_num = 3
    x = []
    for i in range(0, x_num):
        # Random column
        col = np.random.randint(low=4, high=8)
        x_row = []
        go_row = []
        for j in range(0, col):
            x_row.append(np.random.uniform(low=0, high=5, size=(n_i, )))
        x.append(np.asarray(x_row))

    y = np.random.randint(low=0, high=no_softmax, size=x_num)
    nntest = RecurrentNNTest(n_i, no_hidden, no_softmax, use_bias, act_func)

    gc = GradientChecker()
    gc.check_nn(nntest, x, y)


if __name__ == "__main__":
    # feedforward_nntest()
    recurrent_nntest()
