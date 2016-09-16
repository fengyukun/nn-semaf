#! /usr/bin/env python3
"""
Authors: fengyukun
Date:  2016-08-17
Brief:  The implementation of bidirectional recurrent layer
"""

import copy
from inc import*
from gradient_checker import GradientChecker
from layer import Layer
import recurrent_layer
import lstm_layer


class BiRecurrentLayer(Layer):
    """
    Bidirectional recurrent layer class
    """
    def __init__(self):
        pass

    def init_layer(self, n_i, n_o, act_func='tanh',
                   use_bias=True, tfloat='float64', use_lstm=True):
        """
        Initialize parameters of layer
        n_i: int.
            The number of units of the previous layer
        n_o: int.
            The number of units of current layer. Here it is the number of
            blocks
        act_func: str
            Activation function used for cell input and output.
            Two values are tanh and sigmoid
        use_bias: boolean
            Whether to use bias vector on this layer
        tfloat: str
            Type of float used on the weights
        use_lstm: bool
            Whether use lstm layer, default is lstm layer
        """

        self.n_i = n_i
        self.n_o = n_o
        self.act_func = act_func
        self.use_bias = use_bias
        self.tfloat = tfloat
        self.use_lstm = use_lstm

        if self.use_lstm:
            self.upper_layer = lstm_layer.LSTMLayer()
            self.lower_layer = lstm_layer.LSTMLayer()
        else:
            self.upper_layer = recurrent_layer.RecurrentLayer()
            self.lower_layer = recurrent_layer.RecurrentLayer()
        self.upper_layer.init_layer(self.n_i, self.n_o,
                                   act_func=self.act_func,
                                   use_bias=self.use_bias)
        self.lower_layer.share_layer(self.upper_layer)
        self.init_params()

    def init_params(self):
        """ Init params

        """
        self.params = self.upper_layer.params
        self.param_names = self.upper_layer.param_names

        if self.use_lstm:
            self.n_i = self.upper_layer.n_i
            self.n_o = self.upper_layer.n_o
            self.act_func = self.upper_layer.act_func
            self.use_bias = self.upper_layer.use_bias
            self.tfloat = self.upper_layer.tfloat

            self.wxi = self.upper_layer.wxi
            self.wxf = self.upper_layer.wxf
            self.wxc = self.upper_layer.wxc
            self.wxo = self.upper_layer.wxo
            self.whi = self.upper_layer.whi
            self.whc = self.upper_layer.whc
            self.whf = self.upper_layer.whf
            self.who = self.upper_layer.who

            if self.use_bias:
                self.ib = self.upper_layer.ib
                self.fb = self.upper_layer.fb
                self.cb = self.upper_layer.cb
                self.ob = self.upper_layer.ob
        else:
            self.n_i = self.upper_layer.n_i
            self.n_o = self.upper_layer.n_o
            self.act_func = self.upper_layer.act_func
            self.use_bias = self.upper_layer.use_bias
            self.tfloat = self.upper_layer.tfloat

            self.w = self.upper_layer.w
            self.rw = self.upper_layer.rw
            if self.use_bias:
                self.b = self.upper_layer.b

    def share_layer(self, inited_layer):
        """
        Sharing layer with given layer. Parameters are shared with two layers.
        inited_layer: LSTMLayer, RecurrentLayer or BiRecurrentLayer
            inited_layer (initialized layer) is the layer which gives its
            parameters to 'self'.
        """
        self.params = inited_layer.params
        self.param_names = inited_layer.param_names

        if inited_layer.use_lstm:

            self.n_i = inited_layer.n_i
            self.n_o = inited_layer.n_o
            self.act_func = inited_layer.act_func
            self.use_bias = inited_layer.use_bias
            self.tfloat = inited_layer.tfloat

            self.wxi = inited_layer.wxi
            self.wxf = inited_layer.wxf
            self.wxc = inited_layer.wxc
            self.wxo = inited_layer.wxo
            self.whi = inited_layer.whi
            self.whc = inited_layer.whc
            self.whf = inited_layer.whf
            self.who = inited_layer.who

            if self.use_bias:
                self.ib = inited_layer.ib
                self.fb = inited_layer.fb
                self.cb = inited_layer.cb
                self.ob = inited_layer.ob
        else:
            self.n_i = inited_layer.n_i
            self.n_o = inited_layer.n_o
            self.act_func = inited_layer.act_func
            self.use_bias = inited_layer.use_bias
            self.tfloat = inited_layer.tfloat

            self.w = inited_layer.w
            self.rw = inited_layer.rw
            if self.use_bias:
                self.b = inited_layer.b

    def forward(self, x):
        """
        Forward pass.
        x: 3d array-like
            In the whole it usually is jagged array. The first dimension
            is the number of samples. The second is the number of unit
            representation. The third are float numbers in one unit
        --------
        """

        upper_out = self.upper_layer.forward(
            x, starts=None, ends=None, reverse=False, output_opt='full'
        )
        lower_out = self.lower_layer.forward(
            x, starts=None, ends=None, reverse=True, output_opt='full'
        )
        forward_out = copy.deepcopy(upper_out)
        for i in range(0, len(forward_out)):
            for j in range(0, len(forward_out[i])):
                forward_out[i][j] = upper_out[i][j] + lower_out[i][j]
        return forward_out

    def backprop(self, go):
        """
        Back propagation. Note that backprop is only based on the forward pass.
        backprop will choose the lastest forward pass from Multiple forward
        passes.
        go: 3d array-like or 2d numpy array(when output_opt is set to 'last')
            Gradients on the output of current layer.

        output
        --------
        gop: numpy.ndarray
            gradients on output of previous layer. The shape of gop is the
            same as x
        gparams: self.gparams
        """

        # Add the gradients on previous layer together
        upper_gop = self.upper_layer.backprop(go)
        lower_gop = self.lower_layer.backprop(go)
        gop = copy.deepcopy(upper_gop)
        for i in range(0, len(upper_gop)):
            for j in range(0, len(upper_gop[i])):
                gop[i][j] = upper_gop[i][j] + lower_gop[i][j]
    
        # Add the gradients on parameters together
        self.gparams = []
        for i in range(0, len(self.upper_layer.gparams)):
            self.gparams.append(
                self.upper_layer.gparams[i] + self.lower_layer.gparams[i]
            )
        return gop


def layer_test():
    n_i = 3
    n_o = 5
    use_bias = True
    x_num = 5
    x = []
    # Construct x
    for i in range(0, x_num):
        # Random column
        col = np.random.randint(low=3, high=8)
        x_row = []
        for j in range(0, col):
            x_row.append(np.random.uniform(low=0, high=5, size=(n_i, )))
        x.append(np.asarray(x_row))

    birecurrent_layer = BiRecurrentLayer()
    birecurrent_layer.init_layer(n_i=n_i, n_o=n_o, act_func='sigmoid',
                                 use_bias=use_bias, use_lstm=True)

    print(birecurrent_layer.param_names)
    gc = GradientChecker(epsilon=1e-05)
    gc.check_jagged_input(birecurrent_layer, x)
    check_params = None
    gc.check_layer_params(birecurrent_layer, x, check_params)

if __name__ == "__main__":
    layer_test()
