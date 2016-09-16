#! /usr/bin/env python3
"""
Authors: fengyukun
Date:  2016-05-21
Brief:  The implementation of recurrent layer
"""

import copy
from inc import*
from gradient_checker import GradientChecker
from layer import HiddenLayer


class RecurrentLayer(HiddenLayer):
    """
    Recurrent layer class
    """
    def __init__(self):
        self.forward_out = None
        self.x = None

    def init_layer(self, n_i, n_o, act_func='tanh',
                   use_bias=True, tfloat='float64'):
        """
        Initialize parameters of recurrent layer
        n_i: int.
            The number of units of the previous layer
        n_o: int.
            The number of units of current layer
        act_func: str
            Activation function. Two values are tanh and sigmoid
        use_bias: boolean
            Whether to use bias vector on this layer
        tfloat: str
            Type of float used on the weights
        """

        HiddenLayer.__init__(self)
        try:
            HiddenLayer.init_layer(self, n_i, n_o, act_func, use_bias, tfloat)
        except:
            logging.error("Failed to init HiddenLayer")
            raise Exception

    def share_layer(self, recurrent_layer):
        """
        Sharing layer with given layer. Parameters are shared with two layers.
        recurrent_layer: RecurrentLayer
            The parameters of layer to be shared
        """

        self.params = recurrent_layer.params
        self.param_names = recurrent_layer.param_names

        self.n_i = recurrent_layer.n_i
        self.n_o = recurrent_layer.n_o
        self.act_func = recurrent_layer.act_func
        self.use_bias = recurrent_layer.use_bias
        self.tfloat = recurrent_layer.tfloat

        self.w = recurrent_layer.w
        self.rw = recurrent_layer.rw
        if self.use_bias:
            self.b = recurrent_layer.b

    def init_params(self):
        """
        Init parameters
        """

        HiddenLayer.init_params(self)

        # Init recurrent weights
        self.rw = np.random.uniform(
            low=-np.sqrt(3. / (self.n_o)),
            high=np.sqrt(3. / (self.n_o)),
            size=(self.n_o, self.n_o)
        ).astype(dtype=self.tfloat, copy=False)
        if self.act_func == 'sigmoid':
            self.rw *= 4
        self.params.append(self.rw)
        self.param_names.append("rw")

    def forward(self, x, output_opt='full'):
        """
        x: 3d array-like, In the whole it usually is jagged array. The first
        loop is sample numbers. The second is unit representation numbers. The
        third is float numbers in one unit
        output_opt: str
            'full': return full out of all hidden state at all time
            'last': return out of hidden state at last time
        --------
        forward_out: which has the same shape as x
        """

        # Keep track them
        self.x = x
        self.output_opt = output_opt
        # Iterate each sample over x
        self.forward_out = []
        for t in range(0, len(x)):
            # Iterate each unit over x[t]
            hidden_outs = []
            start = 0
            end = len(self.x[t])
            stop = 1
            previous_hidden = np.zeros(shape=self.n_o)
            for i in range(start, end, stop):
                hidden_out = (self.w.dot(np.asarray(x[t][i])) +
                              self.rw.dot(previous_hidden))
                if self.use_bias:
                    hidden_out += self.b
                hidden_out = HiddenLayer.net_input_to_out(self, hidden_out)
                hidden_outs.append(hidden_out)
                previous_hidden = hidden_out
            self.forward_out.append(hidden_outs)
        if self.output_opt == 'full':
            return self.forward_out
        else:
            return get_col_from_jagged_array(-1, self.forward_out)

    def grad_out_to_net_input(self, go, forward_out):
        """
        Computing gradients from output to net input
        go: numpy.ndarray
            Gradients on the output of current layer. The shape of go is
            (dim_unit, )
        """
        # Gradients on net input
        gnet = copy.copy(go)
        if self.act_func == 'tanh':
            gnet = go * (1 - forward_out ** 2)
        else:
            gnet = go * forward_out * (1 - forward_out)

        return gnet

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

        if self.x is None or self.forward_out is None:
            logging.error("No forward pass is computed")
            raise Exception

        gop = copy.copy(self.x)
        self.gw = np.zeros(shape=self.w.shape)
        if self.use_bias:
            self.gb = np.zeros(shape=self.b.shape)
        self.grw = np.zeros(shape=self.rw.shape)
        for t in range(0, len(self.forward_out)):
            previous_grad = np.zeros(shape=(self.n_o, ))
            start = len(self.forward_out[t]) - 1
            end = -1
            stop = -1
            for i in range(start, end, stop):
                gout = previous_grad
                if self.output_opt == 'full' and go[t][i] is not None:
                    gout += np.asarray(go[t][i])
                if (self.output_opt == 'last' and i == start):
                    gout += go[t]

                gnet = self.grad_out_to_net_input(gout, self.forward_out[t][i])
                # Accumulated gradients on parameters
                self.gw += gnet.reshape((self.n_o, 1)).dot(
                    self.x[t][i].reshape(1, self.n_i)
                )
                if i != end + 1:
                    self.grw += gnet.reshape((self.n_o, 1)).dot(
                        self.forward_out[t][i - 1].reshape(1, self.n_o)
                    )
                if self.use_bias:
                    self.gb += gnet
                # Gradients on the previous layer
                gop[t][i] = self.w.T.dot(gnet)
                previous_grad = self.rw.T.dot(gnet)

        self.gparams = [self.gw]
        if self.use_bias:
            self.gparams.append(self.gb)
        self.gparams.append(self.grw)

        return gop


def layer_test():
    n_i = 5
    n_o = 10
    use_bias = True
    x_num = 10
    x = []
    go = []
    # Construct x and go
    for i in range(0, x_num):
        # Random column
        col = np.random.randint(low=1, high=10)
        x_row = []
        go_row = []
        for j in range(0, col):
            x_row.append(np.random.uniform(low=0, high=5, size=(n_i, )))
            go_row.append(np.random.uniform(low=0, high=5, size=(n_o, )))
        x.append(np.asarray(x_row))
        go.append(np.asarray(go_row))

    recurrent_layer = RecurrentLayer()
    recurrent_layer.init_layer(n_i=n_i, n_o=n_o, act_func='tanh',
                               use_bias=use_bias)
    print(recurrent_layer.param_names)
    recurrent_layer.forward(x)
    recurrent_layer.backprop(go)

    gc = GradientChecker(epsilon=1e-05)
    gc.check_layer_params(recurrent_layer, x)
    gc.check_jagged_input(recurrent_layer, x)

if __name__ == "__main__":
    layer_test()
