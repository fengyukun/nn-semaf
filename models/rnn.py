#! /usr/bin/env python3
"""
Authors: fengyukun
Date: 2016-06-10
Brief:  The implementation of Recurrent Neural Network (RNN)
"""

# For python2
from __future__ import print_function
# Activate automatic float divison for python2.
from __future__ import division
import os
import sys
sys.path.append("../lib/")
sys.path.append("../utils/")
from inc import*
from gradient_checker import GradientChecker
import layer
import metrics
import recurrent_layer
import lstm_layer


class RNN(object):
    """
    Recurrent Neural Network (RNN) class
    """
    def init(self, x, label_y, word2vec, n_h, up_wordvec=False,
                 use_bias=True, act_func='tanh', use_lstm=False):
        """
        Init RNN
        x: numpy.ndarray, 2d jagged arry
            The input data. The index of words
        label_y: numpy.ndarray, 1d array
            The right label of x
        word2vec: numpy.ndarray, 2d array
            Each row represents word vectors. E.g.,
            word_vectors = word2vec[word_index]
        n_h: int
            Number of hidden unit
        up_wordvec: boolean
            Whether update word vectors
        use_bias: boolean
            Whether use bias on the layers of nn
        act_func: str
            Activation function in hidden layer.
            Two values are tanh and sigmoid
        use_lstm: bool
            Whether use lstm layer, default is rnn layer
        """

        self.x = x
        self.word2vec = word2vec
        self.up_wordvec = up_wordvec
        self.n_h = n_h
        self.act_func = act_func
        self.use_bias = use_bias
        self.use_lstm = use_lstm

        # label_y should be normalized to continuous integers starting from 0
        self.label_y = label_y
        label_set = set(self.label_y)
        y_set = np.arange(0, len(label_set))
        label_to_y = dict(zip(label_set, y_set))
        self.y = np.array([label_to_y[label] for label in self.label_y])
        self.label_to_y = label_to_y

        # Record the map from label id to label for furthur output
        self.y_to_label = {k: v for v, k in label_to_y.items()}
        self.n_o = y_set.shape[0]   # Number of nn output unit
        # Number of nn input unit
        self.n_i = self.word2vec.shape[1]

        # Init layers
        self.embedding_layer = layer.EmbeddingLayer()
        self.embedding_layer.init_layer(self.word2vec)
        self.layers = []
        self.params = []
        self.param_names = []

        # Init hidden layers
        if self.use_lstm:
            rlayer = lstm_layer.LSTMLayer()
        else:
            rlayer = recurrent_layer.RecurrentLayer()
        rlayer.init_layer(self.n_i, self.n_h,
                                   act_func=self.act_func,
                                   use_bias=self.use_bias)
        self.params += rlayer.params
        self.param_names += rlayer.param_names
        self.layers.append(rlayer)

        # Output layer
        softmax_layer = layer.SoftmaxLayer()
        softmax_layer.init_layer(n_i=self.n_h, n_o=self.n_o,
                                 use_bias=self.use_bias)
        self.params += softmax_layer.params
        self.param_names += softmax_layer.param_names
        self.layers.append(softmax_layer)

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
        print("%s %s %s %s %s %s %s" % (self.n_i, self.n_o, self.act_func, self.use_bias,
              self.use_lstm, self.n_h, self.up_wordvec), file=attributes_file)
        y_to_label = ",".join(['%s:%s' % (k, v) for k, v in self.y_to_label.items()]) 
        print(y_to_label, file=attributes_file)
        layer_names = [neural_layer.__class__.__name__ for neural_layer in self.layers]
        print(" ".join(layer_names), file=attributes_file)
        attributes_file.close()

        # Write paramters to file
        self.embedding_layer.write_to_files("%s/embedding_out.npz" % (target_dir,))
        for neural_layer in self.layers:
            layer_target_dir = "%s/%s" % (target_dir, neural_layer.__class__.__name__)
            neural_layer.write_to_files(layer_target_dir)
            
        logging.info("Finish writting %s layer to %s" % (self.__class__.__name__, target_dir))

    def load_from_files(self, target_dir):
        """Load files to recover one object of this class.

        :target_dir: str, a directory where the attribute file and paramter file are.

        """

        # Load attributes file
        attributes_file = open("%s/attributes.txt" % target_dir, "r")
        try:
            (n_i, n_o, act_func, use_bias, use_lstm, n_h, up_wordvec) = (
                attributes_file.readline().strip().split(" ")
            )
            self.n_i = int(n_i)
            self.n_o = int(n_o)
            self.act_func = act_func
            if use_bias == 'True':
                self.use_bias = True
            else:
                self.use_bias = False
            if use_lstm == 'True':
                self.use_lstm = True
            else:
                self.use_lstm = False
            self.n_h = int(n_h)
            if up_wordvec == 'True':
                self.up_wordvec = True
            else:
                self.up_wordvec = False
            y_to_label = attributes_file.readline().strip().split(',')
            self.y_to_label = {}
            for key_value in y_to_label:
                key, value = key_value.split(":")
                self.y_to_label[int(key)] = value

            layer_names = attributes_file.readline().strip().split(" ")
        except:
            raise Exception("%s/attributes.txt format error" % target_dir)
        attributes_file.close()
        
        # Load parameters file

        self.embedding_layer = layer.EmbeddingLayer()
        self.embedding_layer.load_from_files("%s/embedding_out.npz" % (target_dir,))

        self.layers = []
        self.params = []
        self.param_names = []
        for layer_name in layer_names:
            if layer_name == 'LSTMLayer':
                neural_layer = lstm_layer.LSTMLayer()
            elif layer_name == 'SoftmaxLayer':
                neural_layer = layer.SoftmaxLayer()
            layer_target_dir = "%s/%s" % (target_dir, neural_layer.__class__.__name__)
            neural_layer.load_from_files(layer_target_dir)
            self.layers.append(neural_layer)
            self.params += neural_layer.params
            self.param_names += neural_layer.param_names

        logging.info("Finish loading %s from %s" % (self.__class__.__name__, target_dir))

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
        x: numpy.ndarray, 2d arry
            The input data. The index of words
        """

        layer_out = self.embedding_layer.forward(x, input_opt='jagged')
        for layer in self.layers:
            if (isinstance(layer, recurrent_layer.RecurrentLayer) or
               isinstance(layer, lstm_layer.LSTMLayer)):
                layer_out = layer.forward(layer_out, output_opt='last')
            else:
                layer_out = layer.forward(layer_out)
        self.forward_out = layer_out
        return self.forward_out

    def backprop(self, y):
        """
        Back propagation. Computing gradients on parameters of nn
        y: numpy.ndarray
            Normalized correct label of x
        """

        if not hasattr(self, 'forward_out'):
            logging.error("No forward pass is computed")
            raise Exception

        go = np.zeros(self.forward_out.shape)
        for i in range(0, go.shape[0]):
            go[i][y[i]] = (-1) / self.forward_out[i][y[i]]

        self.gparams = []
        for layer in reversed(self.layers):
            go = layer.backprop(go)
            self.gparams = layer.gparams + self.gparams
        # Gradients on x
        gx = go
        return gx

    def batch_train(self, x, y, lr):
        """
        Batch training on x given right label y
        x: numpy.ndarray, 2d arry
            The input data. The index of words
        y: numpy.ndarray
            Normalized correct label of x
        lr: float
            Learning rate
        """
        self.forward(x)
        gx = self.backprop(y)
        # Update parameters
        for gparam, param in zip(self.gparams, self.params):
            param -= lr * gparam
        if self.up_wordvec:
            (vectorized_x, go) = self.embedding_layer.backprop(gx)
            for i in range(0, len(vectorized_x)):
                for j in range(0, len(vectorized_x[i])):
                    vectorized_x[i][j] -= lr * go[i][j]

    def minibatch_train(self, lr=0.1, minibatch=5, max_epochs=100,
                        verbose=False):
        """
        Minibatch training over x. Training will be stopped when the zero-one
        loss is zero on x.

        lr: float
            Learning rate
        minibatch: int
            Mini batch size
        max_epochs: int
            the max epoch
        verbose: bool
            whether to print information during each epoch training
        Return
        ----
        train_epoch: int
            The epoch number during traing on train data
        """

        for epoch in range(1, max_epochs + 1):
            n_batches = int(self.y.shape[0] / minibatch)
            for batch_i in range(0, n_batches):
                self.batch_train(
                    self.x[batch_i * minibatch:(batch_i + 1) * minibatch],
                    self.y[batch_i * minibatch:(batch_i + 1) * minibatch],
                    lr
                )
            # Train the rest if it has
            if n_batches * minibatch != self.y.shape[0]:
                self.batch_train(
                    self.x[(batch_i + 1) * minibatch:],
                    self.y[(batch_i + 1) * minibatch:],
                    lr
                )
            label_preds = self.predict(self.x)
            error = metrics.zero_one_loss(self.label_y, label_preds)
            cost = self.cost(self.x, self.y)
            if verbose:
                logging.info("epoch: %d training,on train data, "
                             "cross-entropy:%f, zero-one loss: %f"
                             % (epoch, cost, error))
            if abs(error - 0.0) <= 0.001:
                break
        return epoch

    def predict(self, x):
        """
        Prediction of FNN on x

        x: numpy.ndarray, 2d arry
            The input data. The index of words
        Return
        -----
        numpy.ndarray, 1d array. The predict label on x
        """
        py = self.forward(x)
        y = py.argmax(axis=1)
        return np.array([self.y_to_label[i] for i in y])


def rnn_test():
    x_col = 5
    no_softmax = 5
    n_h = 10
    up_wordvec = False
    use_bias = True
    act_func = 'tanh'
    use_lstm = True
    x_row = 5
    voc_size = 20
    word_dim = 3
    x = np.random.randint(low=0, high=voc_size, size=(x_row, x_col))
    label_y = np.random.randint(low=0, high=20, size=x_row)
    word2vec = np.random.uniform(low=0, high=5, size=(voc_size, word_dim))
    nntest = RNN()
    nntest.init(x, label_y, word2vec, n_h, up_wordvec, use_bias,
                act_func, use_lstm=use_lstm)

    # Training
    lr = 0.015
    minibatch = 5
    max_epochs = 100
    verbose = True
    nntest.minibatch_train(lr, minibatch, max_epochs, verbose)
    # Gradient testing
    y = np.array([nntest.label_to_y[i] for i in label_y])
    gc = GradientChecker(epsilon=1e-05)
    gc.check_nn(nntest, x, y)

    # Write and load test
    nntest_bak = RNN()
    nntest.write_to_files("lstm_save")
    nntest_bak.load_from_files("lstm_save")
    print("After loading")
    gc = GradientChecker(epsilon=1e-05)
    gc.check_nn(nntest_bak, x, y)
    print("Orignal output")
    print(nntest.forward(x))
    print("After loading output")
    print(nntest_bak.forward(x))

if __name__ == "__main__":
    rnn_test()
