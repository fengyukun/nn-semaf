#! /usr/bin/env python3
"""
Authors: fengyukun
Date:   2016-05-16
Brief:  The library of layer
"""

# For python2
from __future__ import print_function
import copy
import os
from inc import*
from gradient_checker import GradientChecker


class Layer(object):
    """
    Base layer(empty)
    """
    pass


class NormlizationLayer(Layer):
    """
    Normlization layer class
    """
    def __init__(self, n_unit):
        """
        Init layer
        n_unit: int
            Number of unit
        """
        self.n_unit = n_unit
        self.forward_out = None
        self.x = None

    def backprop(self, go):
        """
        Back propagation. Note that backprop is only based on the forward pass.
        backprop will choose the lastest forward pass from Multiple forward
        passes.
        go: numpy.ndarray
            Gradients on the output of current layer. The shape of go is
            (num_instances, num_outputs)

        output
        --------
        gnet: numpy.ndarray
            Gradients on net the input of current layer.
        """

        if go.shape[1] != self.n_unit:
            logging.error("shape doesn't match, go shape:%s, unit number:%s"
                          % (go, self.n_unit))
            raise Exception
        if self.forward_out is None or self.x is None:
            logging.error("No forward pass")
            raise Exception

        x_sum = self.x.sum(axis=1).reshape((self.x.shape[0], 1))
        gox_sum = (go * self.x).sum(axis=1).reshape((self.x.shape[0], 1))
        gnet = (go * x_sum - gox_sum) / (x_sum ** 2)

        return gnet

    def forward(self, x):
        """
        Compute forward pass
        x: numpy.ndarray
            x is the input data with the shape (num_instances, num_inputs)

        output
        --------
        forward_out: numpy.ndarray
            Output with the shape (num_instances, num_inputs)
        """

        if x.shape[1] != self.n_unit:
            logging.error("input data shape:%s, not match input unit:%s"
                          % (x.shape, self.n_unit))
            raise Exception

        forward_out = x / x.sum(axis=1).reshape((x.shape[0], 1))

        # Keep track of output and input
        self.forward_out = forward_out
        self.x = x

        return forward_out


class FuncNormLayer(NormlizationLayer):
    """
    Normlization layer by using activation function.
    """
    def __init__(self, n_unit, act_func='softmax'):
        """
        Init layer.
        n_unit: int
            Number of unit
        act_func: str
            'softmax': Using softmax function to normalize
            'sigmoid': Using sigmoid function to normalize
        """

        if act_func not in ['softmax', 'sigmoid']:
            logging.error("act_func:%s, not available")
            raise Exception

        NormlizationLayer.__init__(self, n_unit)
        self.act_func = act_func

    def forward(self, x):
        """
        Compute forward pass
        x: numpy.ndarray
            x is the input data with the shape (num_instances,
            num_inputs) or (num_inputs, )

        output
        --------
        forward_out: numpy.ndarray
            Output with the shape (num_instances, num_inputs) or (num_inputs, )
        """

        # Normalize 1d array to 2d array
        self.is_1d = False
        if len(x.shape) == 1:
            self.is_1d = True
            x = x.reshape((1, x.shape[0]))
        if self.act_func == 'softmax':
            # Numerically-stable softmax input
            stable_input = (x - np.max(x, axis=1) .reshape(x.shape[0], 1))
            stable_input = np.exp(stable_input)
        elif self.act_func == 'sigmoid':
            stable_input = sigmoid_array(x)
        else:
            logging.error("Unknown act_func:%s" % (act_func, ))
            raise Exception
        forward_out = NormlizationLayer.forward(self, stable_input)
        if self.is_1d:
            forward_out = forward_out.reshape((forward_out.shape[1], ))

        # Keep track of stable input
        self.stable_input = stable_input
        return forward_out

    def backprop(self, go):
        """
        Back propagation. Note that backprop is only based on the forward pass.
        backprop will choose the lastest forward pass from Multiple forward
        passes.
        go: numpy.ndarray
            Gradients on the output of current layer. The shape of go is
            (num_instances, num_outputs)

        output
        --------
        gnet: numpy.ndarray
            Gradients on net the input of current layer.
        """

        if self.is_1d:
            go = go.reshape((1, go.shape[0]))
        if go.shape[1] != self.n_unit:
            logging.error("shape doesn't match, go shape:%s, unit number:%s"
                          % (go, self.n_unit))
            raise Exception
        if not hasattr(self, 'stable_input'):
            logging.error("No forward pass")
            raise Exception
        gnet = NormlizationLayer.backprop(self, go)
        if self.act_func == 'softmax':
            gnet *= self.stable_input
        elif self.act_func == 'sigmoid':
            gnet *= self.stable_input * (1 - self.stable_input)
        else:
            logging.error("Unknown act_func:%s" % (act_func, ))
            raise Exception
        if self.is_1d:
            gnet = gnet.reshape((gnet.shape[1],))
        return gnet


class GeneralLayer(Layer):
    """
    General layer class.
    """
    def __init__(self):
        # Keep track of lastest forward pass variables
        self.forward_out = None
        self.x = None

    def set_layer(self, w, b=None, use_bias=False):
        """
        Set layer with given params
        w: numpy.ndarry
            The weight of layer which has the shape (n_o, n_i)
        b: numpy.ndarry
            The bias of layer which has the shape (n_o, )
        use_bias: boolean
            Whether to use bias vector on this layer. If b is given and
            use_bias will force to change True. If b is None and use_bias is
            True, the class will init bias
        """
        self.w = w
        self.n_i = w.shape[1]
        self.n_o = w.shape[0]
        self.params = [self.w]
        self.param_names = ['w']
        if b is not None:
            if b.shape[0] != self.n_o:
                logging.error("b is given, but the shape not match w")
                raise Exception
            self.b = b
            self.use_bias = True
        if b is None and use_bias:
            self.use_bias = use_bias
            self.b = np.zeros(shape=self.n_o, dtype=w.dtype)

        if self.use_bias:
            self.params.append(self.b)
            self.param_names.append('b')

    def init_layer(self, n_i, n_o, use_bias=True, tfloat='float64'):
        """
        Initialize parameters of layer.
        n_i: int.
            The number of units of the previous layer
        n_o: int.
            The number of units of current layer
        use_bias: boolean
            Whether to use bias vector on this layer
        tfloat: str
            Type of float used on the weights
        """

        self.n_i = n_i
        self.n_o = n_o
        self.use_bias = use_bias
        self.tfloat = tfloat

        # Init parameters
        self.init_params()

    def init_params(self):
        """
        Init parameters
        """

        self.w = np.random.uniform(
            low=-np.sqrt(1. / self.n_i),
            high=np.sqrt(1. / self.n_i),
            size=(self.n_o, self.n_i)
        ).astype(dtype=self.tfloat, copy=False)

        self.params = [self.w]
        self.param_names = ['w']
        # Init bias on output layer
        if self.use_bias:
            self.b = np.zeros(shape=self.n_o, dtype=self.tfloat)
            self.params.append(self.b)
            self.param_names.append('b')

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
        attributes_file = open("%s/attributes.txt" % target_dir, "w")
        print("%s %s %s %s" % (self.n_i, self.n_o, self.use_bias, self.tfloat),
              file=attributes_file)
        attributes_file.close()
        # Write paramters to file
        if self.use_bias:
            np.savez_compressed("%s/parameters.npz" % target_dir, w=self.w, b=self.b)
        else:
            np.savez_compressed("%s/parameters.npz" % target_dir, w=self.w)
        logging.info("Finish writting %s layer to %s" % (self.__class__.__name__, target_dir))

    def load_from_files(self, target_dir):
        """Load files to recover one object of this class.

        :target_dir: str, a directory where the attribute file and paramter file are.

        """

        logging.info("Start loading %s layer from %s" % (self.__class__.__name__, target_dir))
        # Load attributes file
        attributes_file = open("%s/attributes.txt" % target_dir, "r")
        try:
            (n_i, n_o, use_bias, tfloat) = attributes_file.readline().strip().split(" ")
            self.n_i = int(n_i)
            self.n_o = int(n_o)
            if use_bias == 'True':
                self.use_bias = True
            else:
                self.use_bias = False
            self.tfloat = tfloat
        except:
            raise Exception("%s/attributes.txt format error" % target_dir)
        attributes_file.close()
        
        # Load parameters file
        paramters = np.load("%s/parameters.npz" % target_dir)
        self.w = paramters['w']
        self.params = [self.w]
        self.param_names = ['w']
        if self.use_bias:
            self.b = paramters['b']
            self.params.append(self.b)
            self.param_names.append('b')
        logging.info("Finish loading %s layer from %s" % (self.__class__.__name__, target_dir))

    def forward(self, x):
        """
        Compute forward pass
        x: numpy.ndarray
            x is the input data with the shape (num_instances, num_inputs)

        output
        --------
        forward_out: numpy.ndarray
            Output with the shape (num_instances, num_outputs)
        """

        if x.shape[1] != self.n_i:
            logging.error("input data shape:%s, not match input unit:%s"
                          % (x.shape, self.n_i))
            raise Exception

        net_input = x.dot(self.w.T)
        if self.use_bias:
            net_input += self.b

        try:
            forward_out = self.net_input_to_out(net_input)
        except:
            logging.error("Failed to compute forward out")
            raise Exception

        # Keep track it. This will be used in backprop
        self.forward_out = forward_out
        self.x = x
        return forward_out

    def net_input_to_out(self, net_input):
        """
        Net input to out
        net_input: numpy.ndarray
            Net input
        """

        return np.copy(net_input)

    def grad_out_to_net_input(self, go):
        """
        Computing gradients from output to net input
        go: numpy.ndarray
            Gradients on the output of current layer. The shape of go is
            (num_instances, num_outputs)

        output
        --------
        gradients on the net input
        """

        return np.copy(go)

    def backprop(self, go):
        """
        Back propagation. Note that backprop is only based on the forward pass.
        backprop will choose the lastest forward pass from Multiple forward
        passes.
        go: numpy.ndarray
            Gradients on the output of current layer. The shape of go is
            (num_instances, num_outputs)

        output
        --------
        gop: numpy.ndarray
            gradients on output of previous layer. The shape of gop is
            (num_instances, num_previus_layer_outputs)
        gparams: self.gparams
        """

        if go.shape[1] != self.n_o:
            logging.error("gradients on output shape:%s, "
                          "not match output unit:%s" % (go.shape, self.n_o))
            raise Exception
        if self.forward_out is None or self.x is None:
            logging.error("No forward computing")
            raise Exception
        if self.x.shape[0] != go.shape[0]:
            logging.error("x shape:%s, gradient shape:%s"
                          % (self.x.shape[0], go.shape))

        # Gradients on net input
        gnet = self.grad_out_to_net_input(go)

        # Gradients on the parameters
        self.gparams = []
        gw = gnet.T.dot(self.x)
        self.gparams.append(gw)
        if self.use_bias:
            gb = gnet.sum(axis=0)
            self.gparams.append(gb)

        # Gradients on output of previous layer
        gop = gnet.dot(self.w)
        return gop


class SoftmaxLayer(GeneralLayer):
    """
    Softmax layer class(Numerically-stable)
    """
    def __init__(self):
        GeneralLayer.__init__(self)

    def init_layer(self, n_i, n_o, use_bias=True, tfloat='float64'):
        """
        Initialize parameters of softmax layer
        n_i: int.
            The number of units of the previous layer
        n_o: int.
            The number of units of current layer
        use_bias: boolean
            Whether to use bias vector on this layer
        tfloat: str
            Type of float used on the weights
        """

        GeneralLayer.init_layer(self, n_i, n_o, use_bias, tfloat)

    def net_input_to_out(self, net_input):
        """
        Net input to out. Numerically-stable softmax function
        net_input: numpy.ndarray
            Net input
        """

        forward_out = softmax(net_input)

        return forward_out

    def grad_out_to_net_input(self, go):
        """
        Computing gradients from output to net input
        go: numpy.ndarray
            Gradients on the output of current layer. The shape of go is
            (num_instances, num_outputs)

        output
        --------
        gradients on the net input
        """

        gnet = np.zeros(go.shape)
        for t in range(0, go.shape[0]):
            go_t = go[t]
            gnet_t = gnet[t]
            tmp_sum = (go_t * self.forward_out[t]).sum()
            for i in range(0, go_t.shape[0]):
                gnet_t[i] = self.forward_out[t][i] * (go_t[i] - tmp_sum)

        return gnet


class HiddenLayer(GeneralLayer):
    """
    Hidden layer class
    """
    def __init__(self):
        GeneralLayer.__init__(self)

    def init_layer(self, n_i, n_o, act_func='tanh',
                   use_bias=True, tfloat='float64'):
        """
        Initialize parameters of hidden layer
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

        if act_func not in ['tanh', 'sigmoid']:
            logging.error("act_func:%s, not available" % act_func)
            raise Exception

        self.act_func = act_func
        GeneralLayer.init_layer(self, n_i, n_o, use_bias, tfloat)

    def init_params(self):
        """
        Init parameters
        """

        self.w = np.random.uniform(
            low=-np.sqrt(6. / (self.n_i + self.n_o)),
            high=np.sqrt(6. / (self.n_i + self.n_o)),
            size=(self.n_o, self.n_i)
        ).astype(dtype=self.tfloat, copy=False)
        if self.act_func == 'sigmoid':
            self.w *= 4

        self.params = [self.w]
        self.param_names = ['w']
        # Init bias on output layer
        if self.use_bias:
            self.b = np.zeros(shape=self.n_o, dtype=self.tfloat)
            self.params.append(self.b)
            self.param_names.append('b')

    def grad_out_to_net_input(self, go):
        """
        Computing gradients from output to net input
        go: numpy.ndarray
            Gradients on the output of current layer. The shape of go is
            (num_instances, num_outputs)

        output
        --------
        gradients on the net input
        """

        # Gradients on net input
        gnet = np.copy(go)
        if self.act_func == 'tanh':
            gnet = go * (1 - self.forward_out ** 2)
        else:
            gnet = go * self.forward_out * (1 - self.forward_out)

        return gnet

    def net_input_to_out(self, net_input):
        """
        Net input to out.
        net_input: numpy.ndarray
            Net input
        """

        if self.act_func == 'tanh':
            forward_out = np.tanh(net_input)
        else:
            forward_out = sigmoid_array(net_input)

        return forward_out


class EmbeddingLayer(Layer):
    """
    EmbeddingLayer class
    """
    def __init__(self):
        pass

    def init_layer(self, word2vec):
        """
        word2vec: numpy.ndarray, 2d array
            Word vectors. each row represents word vectors.
            E.g., word_vectors = word2vec[word_index]
        """

        self.word2vec = word2vec

    def write_to_files(self, target_file):
        """Write the word2vec to file

        :target_file: str, the target file

        """

        # Write vectors to file
        np.savez_compressed(target_file, word2vec=self.word2vec)
        logging.info("Finish writting %s layer to %s" % (self.__class__.__name__, target_file))

    def load_from_files(self, target_file):
        """Write the word2vec to file

        :target_file: str, the target file

        """
        # Load parameters file
        paramters = np.load(target_file)
        self.word2vec = paramters['word2vec']
        logging.info("Finish loading %s layer from %s" % (self.__class__.__name__, target_file))

    def forward(self, x, input_opt='regular'):
        """
        Compute forward pass
        x: numpy.ndarray, 2d array or 2d jagged array(input_opt=='jagged')
            Train data. each element in x is word index(int). Word index
            should start from 0 and correspond with row numbers of word2vec.
            one row of  x represents a sentence
        input_opt: str
            'regular': x is 2d numpy array.
            'jagged': x is 2d jagged array.

        Return
        -----------
        forward_out: 2d numpy array if input_opt == 'regular'. If input_opt
        is jagged, output will be 3d array which is used for recurrent layer.

        """

        if input_opt == 'regular':
            vectorized_x = self.word2vec[x].reshape(
                (x.shape[0], self.word2vec.shape[1] * x.shape[1])
            )
        else:
            vectorized_x = []
            for row in x:
                vecx_row = []
                for word_index in row:
                    word_vector = self.word2vec[word_index]
                    vecx_row.append(word_vector)
                vectorized_x.append(vecx_row)

        # Keep track of x
        self.x = x
        self.input_opt = input_opt
        self.vectorized_x = vectorized_x
        return vectorized_x

    def backprop(self, go):
        """
        Backprop pass. Note that backprop is only based on the last forward
        pass.

        go: 3d array-like or 2d numpy array(when input_opt is regular)
            Gradients on the output of current layer.

        Return
        ---------
        if input_opt is 'regular', the word indexs used in forward pass and its
        gradients will be returned. If input_opt is not regular, vectorized_x
        and go will be returned

        """

        if not hasattr(self, 'x'):
            logging.error("No forward pass is computed")
            raise Exception

        if self.input_opt == 'regular':
            word_indexs = []     # word vectors index
            gword_vectors = []   # gradients on vectors
            for row, grow in zip(self.x, go):
                for i in range(0, len(row)):
                    word_index = row[i]
                    # Dimension of word vectors
                    word_dim = self.word2vec.shape[1]
                    gword_vector = grow[i * word_dim:(i + 1) * word_dim]
                    # Accumulate gradients on the same vector
                    if word_index in word_indexs:
                        gword_vectors[word_indexs.index(word_index)] += (
                            gword_vector
                        )
                    else:
                        word_indexs.append(word_index)
                        gword_vectors.append(gword_vector)
            return (word_indexs, gword_vectors)
        else:
            return (self.vectorized_x, go)


class AttentionLayer(Layer):

    """The class of AttentionLayer. The caller must provide one global
    infomation which is used to mix up with the input. Then weighted sum of
    input is returned in the forward pass"""

    def __init__(self, norm_func='softmax'):
        """init layer

        :norm_func: str, two options are 'softmax' and 'sigmoid'

        """
        self.norm_func = norm_func

    def forward(self, x, global_info):
        """Compute forward pass. global_info is used to mix up with x and then
        normalization function is applied. At last the weighted sum is obtained

        :x: 3d jagged array, the length of the first dimension is the number of
        samples. The second is number of unit and the third is the number of
        float value.
        :global_info: 2d array, numpy.ndarray
        :returns: 
            - weighted sum, the same shape with global_info
            - attention matrix        

        """

        self.norm_layers = []
        # Numerical value after normalization
        self.after_norm_vals = []
        weighted_sums = np.zeros(shape=(len(x), len(global_info[0])))
        for i in range(0, len(x)):
            row_len = len(x[i])
            # Numerical value before normalization
            before_norm_val = np.zeros(shape=(row_len,))
            for j in range(0, row_len):
                before_norm_val[j] = (
                    x[i][j].dot(global_info[i])
                )
            self.norm_layers.append(
                FuncNormLayer(row_len, act_func=self.norm_func)
            )
            after_norm_val = self.norm_layers[i].forward(before_norm_val)
            self.after_norm_vals.append(after_norm_val)
            # Compute weighted sum
            for j in range(0, row_len):
                weighted_sums[i] += (x[i][j] * after_norm_val[j])

        self.x = x
        self.global_info = global_info
        return (weighted_sums, self.after_norm_vals)

    def backprop(self, go):
        """Backprop pass

        :go: gradient on the output of forward pass.
        :returns: gradient on x and global_info

        """

        # Compute gradients on before_norm_val and on x
        gx = copy.deepcopy(self.x)
        set_jagged_array(gx, 0)
        gglobal_info = np.zeros(shape=self.global_info.shape)
        gbefore_norm_vals = copy.deepcopy(self.after_norm_vals)
        for i in range(0, len(gbefore_norm_vals)):
            row_len = len(self.x[i])

            # Compute part of graidents on x and before_norm_val
            for j in range(0, row_len):
                gx[i][j] = (go[i] * self.after_norm_vals[i][j])
                gbefore_norm_vals[i][j] = (
                    go[i].dot(self.x[i][j])
                )

            gbefore_norm_vals[i] = (
                self.norm_layers[i].backprop(gbefore_norm_vals[i])
            )

            # Compute another part gradients on x and global_info
            for j in range(0, row_len):
                gx[i][j] += (self.global_info[i] * gbefore_norm_vals[i][j])
                gglobal_info[i] += self.x[i][j] * gbefore_norm_vals[i][j]

        return (gx, gglobal_info)

def layer_test():
    n_i = 5
    n_o = 10
    use_bias = True
    x_num = 1
    x = np.random.uniform(low=0, high=5, size=(x_num, n_i))

    softmax_layer = SoftmaxLayer()
    softmax_layer.init_layer(n_i=n_i, n_o=n_o, use_bias=use_bias)
    hidden_layer = HiddenLayer()
    hidden_layer.init_layer(n_i=n_i, n_o=n_o, act_func='sigmoid',
                            use_bias=use_bias)
    norm_layer = NormlizationLayer(x.shape[1])
    funcnorm_layer = FuncNormLayer(x.shape[1], act_func='softmax')
    general_layer = GeneralLayer()
    general_layer.init_layer(n_i=n_i, n_o=n_o, use_bias=use_bias)
    general_layer_list = [softmax_layer, hidden_layer, general_layer]
    norm_layer_list = [norm_layer, funcnorm_layer]

    gc = GradientChecker()

    for layer in general_layer_list:
        gc.check_layer_params(layer, x)
        gc.check_layer_input(layer, x)
        logging.info("")

    for layer in norm_layer_list:
        gc.check_layer_input(layer, x)

    # EmbeddingLayer logic test
    embedding_layer = EmbeddingLayer()

    word2vec_size = (10, 5)
    embedding_x_size = (word2vec_size[0], 4)
    word2vec = np.random.uniform(-4, 4, size=word2vec_size)
    embedding_x = np.random.randint(0, word2vec_size[0], size=embedding_x_size)
    embedding_layer.init_layer(word2vec)
    vectorized_x = embedding_layer.forward(embedding_x, input_opt='jagged')
    embedding_layer.backprop(vectorized_x)

    # Write and load test
    #  softmax_layer.write_to_files("softmax_layer_dir")
    #  softmax_layer_bak = SoftmaxLayer()
    #  softmax_layer_bak.load_from_files("softmax_layer_dir")
    #  print("Checking the loaded layer")
    #  gc.check_layer_params(softmax_layer_bak, x)
    #  gc.check_layer_input(softmax_layer_bak, x)
    #  print("Original layer output:")
    #  print(softmax_layer.forward(x))
    #  print("After loading layer output:")
    #  print(softmax_layer_bak.forward(x))

    # Write and load test embedding layer
    #  embedding_layer.write_to_files("embedding_out.npz")
    #  embedding_layer_bak = EmbeddingLayer()
    #  embedding_layer_bak.load_from_files("embedding_out.npz")
    #  print("Original embedding layer output:")
    #  print(vectorized_x)
    #  print("After loading layer output:")
    #  print(embedding_layer_bak.forward(embedding_x, input_opt='jagged'))



if __name__ == "__main__":
    layer_test()
