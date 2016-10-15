#! /usr/bin/env python3
"""
Authors: fengyukun
Date:  2016-05-26
Brief:  The implementation of long short-term memory (LSTM) layer
"""

# For python2
from __future__ import print_function
import copy
import os
from inc import*
from gradient_checker import GradientChecker
from layer import Layer


class LSTMLayer(Layer):
    """
    The long short-term memory (LSTM) layer class
    """
    def __init__(self):
        pass

    def init_layer(self, n_i, n_o, act_func='tanh',
                   use_bias=True, tfloat='float64'):
        """
        Initialize parameters of LSTM layer
        n_i: int.
            The number of units of the previous layer
        n_o: int.
            The number of units of current layer. Here it it the number of
            blocks
        act_func: str
            Activation function used for cell input and output.
            Two values are tanh and sigmoid
        use_bias: boolean
            Whether to use bias vector on this layer
        tfloat: str
            Type of float used on the weights
        """

        self.n_i = n_i
        self.n_o = n_o
        self.act_func = act_func
        self.use_bias = use_bias
        self.tfloat = tfloat
        self.init_params()

    def write_to_files(self, target_dir):
        """Write the attributes and the parameters to files

        :target_dir: str, a directory where the attribute file and paramter file are. A directory
        will be created if the target_dir does not exist.

        """

        try:
            os.makedirs(target_dir)
        except:
            if not os.path.isdir(target_dir):
                raise Exception("%s is not a directory" % (target_dir,))

        # Write the attributes to file
        logging.info("Begin writting lstm layer to %s" % target_dir)
        attributes_file = open("%s/attributes.txt" % target_dir, "w")
        print("%s %s %s %s %s" % (self.n_i, self.n_o, self.act_func, self.use_bias, self.tfloat), file=attributes_file)
        attributes_file.close()

        # Write paramters to file
        if self.use_bias:
            np.savez_compressed("%s/parameters.npz" % target_dir, wxi=self.wxi, wxf=self.wxf,
                                wxc=self.wxc, wxo=self.wxo, whi=self.whi, whc=self.whc,
                                whf=self.whf, who=self.who, ib=self.ib, fb=self.fb, cb=self.cb,
                                ob=self.ob)
        else:
            np.savez_compressed("%s/parameters.npz" % target_dir, wxi=self.wxi, wxf=self.wxf,
                                wxc=self.wxc, wxo=self.wxo, whi=self.whi, whc=self.whc,
                                whf=self.whf, who=self.who)
        logging.info("End writting lstm layer to %s" % target_dir)

    def load_from_files(self, target_dir):
        """Load files to recover one object of this class.

        :target_dir: str, a directory where the attribute file and paramter file are.

        """

        logging.info("Begin loading lstm layer from %s" % target_dir)
        # Load attributes file
        attributes_file = open("%s/attributes.txt" % target_dir, "r")
        try:
            (n_i, n_o, act_func, use_bias, tfloat) = attributes_file.readline().strip().split(" ")
            self.n_i = int(n_i)
            self.n_o = int(n_o)
            self.act_func = act_func
            self.use_bias = bool(use_bias)
            self.tfloat = tfloat
        except:
            raise Exception("%s/attributes.txt format error" % target_dir)
        attributes_file.close()
        
        # Load parameters file
        paramters = np.load("%s/parameters.npz" % target_dir)
        self.wxi = paramters['wxi']
        self.wxf = paramters['wxf']
        self.wxc = paramters['wxc']
        self.wxo = paramters['wxo']
        self.whi = paramters['whi']
        self.whc = paramters['whc']
        self.whf = paramters['whf']
        self.who = paramters['who']
        if self.use_bias:
            self.ib = paramters['ib']
            self.fb = paramters['fb']
            self.cb = paramters['cb']
            self.ob = paramters['ob']

        self.params = []
        self.param_names = []
        self.params.append(self.wxi)
        self.param_names.append("wxi")
        if self.use_bias:
            self.params.append(self.ib)
            self.param_names.append("ib")
        self.params.append(self.wxf)
        self.param_names.append("wxf")
        if self.use_bias:
            self.params.append(self.fb)
            self.param_names.append("fb")
        self.params.append(self.wxc)
        self.param_names.append("wxc")
        if self.use_bias:
            self.params.append(self.cb)
            self.param_names.append("cb")
        self.params.append(self.wxo)
        self.param_names.append("wxo")
        if self.use_bias:
            self.params.append(self.ob)
            self.param_names.append("ob")
        self.params.append(self.whi)
        self.param_names.append("whi")
        self.params.append(self.whc)
        self.param_names.append("whc")
        self.params.append(self.whf)
        self.param_names.append("whf")
        self.params.append(self.who)
        self.param_names.append("who")

        logging.info("End loading lstm layer from %s" % target_dir)

    def share_layer(self, lstm_layer):
        """
        Sharing layer with given layer. Parameters are shared with two layers.
        lstm_layer: LSTMLayer
            The parameters of layer to be shared
        """

        self.params = lstm_layer.params
        self.param_names = lstm_layer.param_names

        self.n_i = lstm_layer.n_i
        self.n_o = lstm_layer.n_o
        self.act_func = lstm_layer.act_func
        self.use_bias = lstm_layer.use_bias
        self.tfloat = lstm_layer.tfloat

        self.wxi = lstm_layer.wxi
        self.wxf = lstm_layer.wxf
        self.wxc = lstm_layer.wxc
        self.wxo = lstm_layer.wxo
        self.whi = lstm_layer.whi
        self.whc = lstm_layer.whc
        self.whf = lstm_layer.whf
        self.who = lstm_layer.who

        if self.use_bias:
            self.ib = lstm_layer.ib
            self.fb = lstm_layer.fb
            self.cb = lstm_layer.cb
            self.ob = lstm_layer.ob


    def init_params(self):
        """
        Init parameters
        """

        self.params = []
        self.param_names = []

        # Weigths between input x and input gates
        self.wxi = np.random.uniform(
            low=-np.sqrt(1. / self.n_i),
            high=np.sqrt(1. / self.n_i),
            size=(self.n_o, self.n_i)
        ).astype(dtype=self.tfloat, copy=False)
        self.params.append(self.wxi)
        self.param_names.append("wxi")
        if self.use_bias:
            self.ib = np.zeros(shape=self.n_o, dtype=self.tfloat)
            self.params.append(self.ib)
            self.param_names.append("ib")

        # Weights between input x and forget gates
        self.wxf = np.random.uniform(
            low=-np.sqrt(1. / self.n_i),
            high=np.sqrt(1. / self.n_i),
            size=(self.n_o, self.n_i)
        ).astype(dtype=self.tfloat, copy=False)
        self.params.append(self.wxf)
        self.param_names.append("wxf")
        if self.use_bias:
            self.fb = np.zeros(shape=self.n_o, dtype=self.tfloat)
            self.params.append(self.fb)
            self.param_names.append("fb")

        # Weigths between input x and cell
        self.wxc = np.random.uniform(
            low=-np.sqrt(1. / self.n_i),
            high=np.sqrt(1. / self.n_i),
            size=(self.n_o, self.n_i)
        ).astype(dtype=self.tfloat, copy=False)
        self.params.append(self.wxc)
        self.param_names.append("wxc")
        if self.use_bias:
            self.cb = np.zeros(shape=self.n_o, dtype=self.tfloat)
            self.params.append(self.cb)
            self.param_names.append("cb")

        # Weigths between input x and output gates
        self.wxo = np.random.uniform(
            low=-np.sqrt(1. / self.n_i),
            high=np.sqrt(1. / self.n_i),
            size=(self.n_o, self.n_i)
        ).astype(dtype=self.tfloat, copy=False)
        self.params.append(self.wxo)
        self.param_names.append("wxo")
        if self.use_bias:
            self.ob = np.zeros(shape=self.n_o, dtype=self.tfloat)
            self.params.append(self.ob)
            self.param_names.append("ob")

        # Recurrent weights init

        # Weights between previous hidden output and input gates
        self.whi = np.random.uniform(
            low=-np.sqrt(1. / self.n_o),
            high=np.sqrt(1. / self.n_o),
            size=(self.n_o, self.n_o)
        ).astype(dtype=self.tfloat, copy=False)
        self.params.append(self.whi)
        self.param_names.append("whi")

        # Weights between previous hidden output and cell
        self.whc = np.random.uniform(
            low=-np.sqrt(1. / self.n_o),
            high=np.sqrt(1. / self.n_o),
            size=(self.n_o, self.n_o)
        ).astype(dtype=self.tfloat, copy=False)
        self.params.append(self.whc)
        self.param_names.append("whc")

        # Weights between previous hidden output and forget gates
        self.whf = np.random.uniform(
            low=-np.sqrt(1. / self.n_o),
            high=np.sqrt(1. / self.n_o),
            size=(self.n_o, self.n_o)
        ).astype(dtype=self.tfloat, copy=False)
        self.params.append(self.whf)
        self.param_names.append("whf")

        # Weights between previous hidden output and output gates
        self.who = np.random.uniform(
            low=-np.sqrt(1. / self.n_o),
            high=np.sqrt(1. / self.n_o),
            size=(self.n_o, self.n_o)
        ).astype(dtype=self.tfloat, copy=False)
        self.params.append(self.who)
        self.param_names.append("who")

    def set_layer(self):
        """
        Set layer with given params
        """
        pass

    def single_forward(self, x_t, ht_1, ct_1):
        """
        Computing forward in a single pass at time t.
        x_t: numpy.ndarray
            The input data whose shape is (self.n_i, )
        ht_1: numpy.ndarray
            The output of hidden at t - 1. The shape is (self.n_o, )
        ct_1: numpy.ndarray
            The output of cell at t - 1. The shape is (self.n_o, )

        Return
        ---------
        ct: numpy.ndarray
            The output of cell at time t
        ot: numpy.ndarray
            The output of output gates at time t
        scaled_oct: numpy.ndarray
            Scaled output of cell at time t
        scaled_incellt: numpy.ndarray
            Scaled input of cell at time t
        it: numpy.ndarray
            The output of input gates at time t
        xt: numpy.ndarray
            The input of blocks at time t
        ht: numpy.ndarray
            The output of blocks at time t
        ft: numpy.ndarray
            The output of forget gates at time t
        """

        # Input gate at time t
        it = self.wxi.dot(x_t) + self.whi.dot(ht_1)
        if self.use_bias:
            it += self.ib
        it = sigmoid_array(it)

        # Forget gate at time t
        ft = self.wxf.dot(x_t) + self.whf.dot(ht_1)
        if self.use_bias:
            ft += self.fb
        ft = sigmoid_array(ft)

        # Cell output at time t
        scaled_incellt = self.wxc.dot(x_t) + self.whc.dot(ht_1)
        if self.use_bias:
            scaled_incellt += self.cb
        if self.act_func == 'tanh':
            scaled_incellt = np.tanh(scaled_incellt)
        else:
            scaled_incellt = sigmoid_array(scaled_incellt)
        ct = it * scaled_incellt + ct_1 * ft

        # Output gate
        ot = self.wxo.dot(x_t) + self.who.dot(ht_1)
        if self.use_bias:
            ot += self.ob
        ot = sigmoid_array(ot)

        if self.act_func == 'tanh':
            scaled_oct = np.tanh(ct)
        else:
            scaled_oct = sigmoid_array(ct)
        ht = ot * scaled_oct

        return (ct, ot, scaled_oct, scaled_incellt, it, ht, ft)

    def single_backprop(self, ght, gct, ct, ot, ct_1, scaled_oct,
                        scaled_incellt, it, xt, ht_1, ft):
        """
        Backprop in a single pass at time t.
        ght: numpy.ndarray
            Accumulated gradients on output of blocks at time t
        gct: numpy.ndarray
            Accumulated gradients on output of cell at time t
        ct: numpy.ndarray
            The output of cell at time t
        ot: numpy.ndarray
            The output of output gates at time t
        ct_1: numpy.ndarray
            The output of cell at time t - 1
        scaled_oct: numpy.ndarray
            Scaled output of cell at time t
        scaled_incellt: numpy.ndarray
            Scaled input of cell at time t
        it: numpy.ndarray
            The output of input gates at time t
        xt: numpy.ndarray
            The input of blocks at time t
        ht_1: numpy.ndarray
            The output of blocks at time t - 1
        ft: numpy.ndarray
            The output of forget gates at time t

        Returns
        ------------
        gxt: numpy.ndarray
            Gradients on xt
        ght_1: numpy.ndarray
            Gradients on output of blocks at t - 1
        gct_1: numpy.ndarray
            Gradients on output of cell at t - 1
        """

        # Gradients on output of cell from top of blocks
        gcell = ght * ot
        if self.act_func == 'tanh':
            gcell *= (1 - scaled_oct ** 2)
        else:
            gcell *= scaled_oct * (1 - scaled_oct)
        # Accumulated gradients on cell
        gcell += gct
        # Gradients on cell at t - 1
        gct_1 = gcell * ft

        # Gradients on output gates
        gogates = ght * scaled_oct
        # Gradients on forget gates
        gfgates = gcell * ct_1
        # Gradients on input gates
        gigates = gcell * scaled_incellt
        # Gradients on input of cell
        gincell = gcell * it
        if self.act_func == 'tanh':
            gincell *= (1 - scaled_incellt ** 2)
        else:
            gincell *= scaled_incellt * (1 - scaled_incellt)

        # Gradients on input of output gates
        giogates = gogates * ot * (1 - ot)
        # Gradients on wxo, who and ob
        self.gwxo += giogates.reshape((self.n_o, 1)).dot(
            xt.reshape((1, self.n_i))
        )
        self.gwho += giogates.reshape((self.n_o, 1)).dot(
            ht_1.reshape((1, self.n_o))
        )
        if self.use_bias:
            self.gob += giogates
        # Gradients on xt
        gxt = 0
        gxt += self.wxo.T.dot(giogates)
        # Gradients on previous output of blocks
        ght_1 = 0
        ght_1 += self.who.T.dot(giogates)

        # Gradients on input of forget gates
        gifgates = gfgates * ft * (1 - ft)
        # Gradients on wxf, whf and fb
        self.gwxf += gifgates.reshape((self.n_o, 1)).dot(
            xt.reshape((1, self.n_i))
        )
        self.gwhf += gifgates.reshape((self.n_o, 1)).dot(
            ht_1.reshape((1, self.n_o))
        )
        if self.use_bias:
            self.gfb += gifgates
        # Gradients on xt
        gxt += self.wxf.T.dot(gifgates)
        # Gradients on previous output of blocks
        ght_1 += self.whf.T.dot(gifgates)

        # Gradients on input of input gates
        giigates = gigates * it * (1 - it)
        # Gradients on wxi, whi and ib
        self.gwxi += giigates.reshape((self.n_o, 1)).dot(
            xt.reshape((1, self.n_i))
        )
        self.gwhi += giigates.reshape((self.n_o, 1)).dot(
            ht_1.reshape((1, self.n_o))
        )
        if self.use_bias:
            self.gib += giigates
        # Gradients on xt
        gxt += self.wxi.T.dot(giigates)
        ght_1 += self.whi.T.dot(giigates)

        # Gradients on input of cell
        self.gwxc += gincell.reshape((self.n_o, 1)).dot(
            xt.reshape((1, self.n_i))
        )
        self.gwhc += gincell.reshape((self.n_o, 1)).dot(
            ht_1.reshape((1, self.n_o))
        )
        if self.use_bias:
            self.gcb += gincell
        # Gradients on xt
        gxt += self.wxc.T.dot(gincell)
        ght_1 += self.whc.T.dot(gincell)

        return (gxt, ght_1, gct_1)

    def forward(self, x, starts=None, ends=None, reverse=False, output_opt='full'):
        """
        Forward pass on [start, end) of each row in x.
        x: 3d array-like
            In the whole it usually is jagged array. The first dimension
            is the number of samples. The second is the number of unit
            representation. The third are float numbers in one unit
        starts: list of int
            Start positions (included) in each row of x. Its default value
            None means zeros.
        ends: list of int
            End position (not included) in each row of x. Its default value
            None represents the last position plus 1 (not included) in each
            row of x.
        reverse: boolean
            False: keep the order in [start, end)
            True: keep in a reverse order (end, start]
        output_opt: str
            'full': return full out of all blocks at all time
            'last': return out of all blocks at last time
        --------
        """

        # Keep track them
        self.x = x
        self.output_opt = output_opt
        self.starts = starts
        self.ends = ends
        self.reverse = reverse
        # The offset positions which are the left positions of the selected
        # interval in the row of x.
        if self.starts is None:
            self.offsets_pos = [0 for i in range(0, len(x))]
        else:
            self.offsets_pos = self.starts

        self.cts = []
        self.ots = []
        self.scaled_octs = []
        self.scaled_incellts = []
        self.its = []
        self.hts = []
        self.fts = []

        # Iterate each sample over x
        for i in range(0, len(x)):
            cts = []
            ots = []
            scaled_octs = []
            scaled_incellts = []
            its = []
            hts = []
            fts = []
            # Iterate each unit over x[i]
            ht_1 = np.zeros((self.n_o, ))
            ct_1 = np.zeros((self.n_o, ))
            if self.ends is None:
                end = len(self.x[i])
            else:
                end = self.ends[i]
            if self.starts is None:
                start = 0
            else:
                start = self.starts[i]
            sub_x = x[i][start:end]
            if self.reverse == True:
                sub_x = reversed(sub_x)
            for x_t in sub_x:
                (ct, ot, scaled_oct, scaled_incellt, it, ht, ft) = (
                    self.single_forward(np.asarray(x_t), ht_1, ct_1)
                )
                ht_1 = ht
                ct_1 = ct
                cts.append(ct)
                ots.append(ot)
                scaled_octs.append(scaled_oct)
                scaled_incellts.append(scaled_incellt)
                its.append(it)
                hts.append(ht)
                fts.append(ft)

            # Keep track middle varibles
            self.cts.append(cts)
            self.ots.append(ots)
            self.scaled_octs.append(scaled_octs)
            self.scaled_incellts.append(scaled_incellts)
            self.its.append(its)
            self.hts.append(hts)
            self.fts.append(fts)

        if self.output_opt == 'full':
            if self.reverse:
                return inverse_jagged_array(self.hts)
            else:
                return self.hts
        else:
            return get_col_from_jagged_array(-1, self.hts)

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

        if not hasattr(self, 'x'):
            logging.error("No forward pass is computed")
            raise Exception

        # Only in reverse and 'full' model, go need be inversed
        # In 'last' mode, go is 2d array and do not need be inversed
        if self.reverse and self.output_opt == 'full':
            go = inverse_jagged_array(go)
        gop = copy.deepcopy(self.hts)

        # Init gradients on parameters
        self.gparams = []
        self.gwxi = np.zeros(self.wxi.shape)
        self.gparams.append(self.gwxi)
        if self.use_bias:
            self.gib = np.zeros(self.ib.shape)
            self.gparams.append(self.gib)

        self.gwxf = np.zeros(self.wxf.shape)
        self.gparams.append(self.gwxf)
        if self.use_bias:
            self.gfb = np.zeros(self.fb.shape)
            self.gparams.append(self.gfb)

        self.gwxc = np.zeros(self.wxc.shape)
        self.gparams.append(self.gwxc)
        if self.use_bias:
            self.gcb = np.zeros(self.cb.shape)
            self.gparams.append(self.gcb)

        self.gwxo = np.zeros(self.wxo.shape)
        self.gparams.append(self.gwxo)
        if self.use_bias:
            self.gob = np.zeros(self.ob.shape)
            self.gparams.append(self.gob)

        # Gradients on recurrent weights
        self.gwhi = np.zeros(self.whi.shape)
        self.gparams.append(self.gwhi)
        self.gwhc = np.zeros(self.whc.shape)
        self.gparams.append(self.gwhc)
        self.gwhf = np.zeros(self.whf.shape)
        self.gparams.append(self.gwhf)
        self.gwho = np.zeros(self.who.shape)
        self.gparams.append(self.gwho)

        for i in range(0, len(self.hts)):
            ght = np.zeros((self.n_o, ))
            gct = np.zeros((self.n_o, ))
            start = len(self.hts[i]) - 1
            end = -1
            stop = -1
            for t in range(start, end, stop):
                if self.output_opt == 'full' and go[i][t] is not None:
                    ght += go[i][t]
                if self.output_opt == 'last' and t == start:
                    ght += go[i]
                if t == end + 1:
                    ht_1 = np.zeros((self.n_o, ))
                    ct_1 = np.zeros((self.n_o, ))
                else:
                    ht_1 = self.hts[i][t - 1]
                    ct_1 = self.cts[i][t - 1]
                if self.reverse == False:
                    x_pos = t + self.offsets_pos[i]
                else:
                    x_pos = self.offsets_pos[i] + len(self.hts[i]) - 1 - t
                (gxt, ght_1, gct_1) = self.single_backprop(
                    ght, gct, self.cts[i][t], self.ots[i][t],
                    ct_1, self.scaled_octs[i][t],
                    self.scaled_incellts[i][t], self.its[i][t],
                    self.x[i][x_pos], ht_1, self.fts[i][t]
                )
                if self.reverse:
                    pos = len(self.hts[i]) - 1 - t
                    gop[i][pos] = gxt
                else:
                    gop[i][t] = gxt

                ght = ght_1
                gct = gct_1

        return gop

def layer_test():
    n_i = 3
    n_o = 5
    use_bias = True
    x_num = 3
    x = []
    go = []
    # Construct x and go
    for i in range(0, x_num):
        # Random column
        col = np.random.randint(low=1, high=4)
        x_row = []
        go_row = []
        for j in range(0, col):
            x_row.append(np.random.uniform(low=0, high=5, size=(n_i, )))
            go_row.append(np.random.uniform(low=0, high=5, size=(n_o, )))
        x.append(np.asarray(x_row))
        go.append(np.asarray(go_row))

    lstm_layer = LSTMLayer()
    lstm_layer.init_layer(n_i=n_i, n_o=n_o, act_func='sigmoid',
                          use_bias=use_bias)

    print(lstm_layer.param_names)
    gc = GradientChecker(epsilon=1e-05)
    gc.check_jagged_input(lstm_layer, x)
    check_params = None
    gc.check_layer_params(lstm_layer, x, check_params)

    # Write and load test
    #  lstm_layer.write_to_files("lstm_layer_dir")
    #  lstm_layer_bak = LSTMLayer()
    #  lstm_layer_bak.load_from_files("lstm_layer_dir")
    #  print("After writting and loading")
    #  print(lstm_layer.param_names)
    #  gc = GradientChecker(epsilon=1e-05)
    #  gc.check_jagged_input(lstm_layer, x)
    #  check_params = None
    #  gc.check_layer_params(lstm_layer, x, check_params)
    #  print("orginal output:")
    #  print(lstm_layer.forward(x)[0][0:2])
    #  print("after loading output:")
    #  print(lstm_layer_bak.forward(x)[0][0:2])

if __name__ == "__main__":
    layer_test()
