#! /usr/bin/env python3
"""
Authors: fengyukun
Date: 2016-06-18
Brief:  The implementation of Bidirectional Recurrent Neural Network (BRNN)
"""

import sys
sys.path.append("../lib/")
sys.path.append("../utils/")
from inc import*
from gradient_checker import GradientChecker
import layer
import metrics
import recurrent_layer
import lstm_layer


class BRNN(object):
    """
    Bidirectional Recurrent Neural Network (BRNN) class
    """
    def __init__(self, x, label_y, word2vec, n_h, up_wordvec=False,
                 use_bias=True, act_func='tanh', use_lstm=False):
        """
        Init BRNN
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
            self.left_layer = lstm_layer.LSTMLayer()
            self.right_layer = lstm_layer.LSTMLayer()
        else:
            self.left_layer = recurrent_layer.RecurrentLayer()
            self.right_layer = recurrent_layer.RecurrentLayer()
        self.left_layer.init_layer(self.n_i, self.n_h,
                                   act_func=self.act_func,
                                   use_bias=self.use_bias)
        self.right_layer.share_layer(self.left_layer)

        self.params += self.left_layer.params
        self.param_names += self.left_layer.param_names

        # Output layer
        self.softmax_layer = layer.SoftmaxLayer()
        self.softmax_layer.init_layer(n_i=self.n_h, n_o=self.n_o,
                                 use_bias=self.use_bias)
        self.params += self.softmax_layer.params
        self.param_names += self.softmax_layer.param_names

    def cost(self, x, y, split_pos=None):
        """
        Cost function
        split_pos: 1d array like
            Start position in x. BRNN will compute from split_pos. The
            left_layer will compute from 0 to split_pos(not included)
            and the right_layer will compute from split_pos to last.
            If split_pos is None, split_pos will
            be the half of current row of x.
        """
        py = self.forward(x, split_pos)
        cross_entropy = -np.sum(
            np.log(py[np.arange(0, y.shape[0]), y])
        )
        return cross_entropy

    def forward(self, x, split_pos=None):
        """
        Compute forward pass
        x: numpy.ndarray, 2d arry
            The input data. The index of words
        split_pos: 1d array like
            Start position in x. BRNN will compute from split_pos. The
            left_layer will compute from 0 to split_pos(not included)
            and the right_layer will compute from split_pos to last.
            If split_pos is None, split_pos will
            be the half of current row of x.
        """

        embedding_out = self.embedding_layer.forward(x, input_opt='jagged')

        ends_for_left = None
        if split_pos is not None:
            ends_for_left = ends=[x + 1 for x in split_pos]
        left_out = self.left_layer.forward(
            embedding_out, starts=None, ends=split_pos,
            reverse=False, output_opt='last'
        )
        right_out = self.right_layer.forward(
            embedding_out, starts=split_pos, ends=None,
            reverse=True, output_opt='last'
        )
        # Avoiding empty output due to empty input caused by spliting
        recurrent_out = add_two_array(left_out, right_out)
        self.forward_out = self.softmax_layer.forward(recurrent_out)
        self.split_pos = split_pos

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

        go = self.softmax_layer.backprop(go)
        self.gparams = []
        self.gparams = self.softmax_layer.gparams + self.gparams
        gx = merge_jagged_array(
            self.left_layer.backprop(go),
            self.right_layer.backprop(go)
        )
        recurrent_gparams = []
        for i in range(0, len(self.left_layer.gparams)):
            recurrent_gparams.append(
                self.left_layer.gparams[i] + self.right_layer.gparams[i]
            )
        self.gparams = recurrent_gparams + self.gparams
        return gx

    def batch_train(self, x, y, lr, split_pos):
        """
        Batch training on x given right label y
        x: numpy.ndarray, 2d arry
            The input data. The index of words
        y: numpy.ndarray
            Normalized correct label of x
        lr: float
            Learning rate
        split_pos: 1d array like
            Start position in x. BRNN will compute from split_pos. The
            left_layer will compute from 0 to split_pos(not included)
            and the right_layer will compute from split_pos to last.
            If split_pos is None, split_pos will
            be the half of current row of x.
        """
        self.forward(x, split_pos)
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
                        split_pos=None, verbose=False):
        """
        Minibatch training over x. Training will be stopped when the zero-one
        loss is zero on x.

        lr: float
            Learning rate
        minibatch: int
            Mini batch size
        max_epochs: int
            the max epoch
        split_pos: 1d array like
            Start position in x. BRNN will compute from split_pos. The
            left_layer will compute from 0 to split_pos(not included)
            and the right_layer will compute from split_pos to last.
            If split_pos is None, split_pos will
            be the half of current row of x.
        verbose: bool
            whether to print information during each epoch training
        Return
        ----
        train_epoch: int
            The epoch number during traing on train data
        """

        for epoch in range(1, max_epochs + 1):
            n_batches = int(self.y.shape[0] / minibatch)
            batch_i = 0
            for batch_i in range(0, n_batches):
                self.batch_train(
                    self.x[batch_i * minibatch:(batch_i + 1) * minibatch],
                    self.y[batch_i * minibatch:(batch_i + 1) * minibatch],
                    lr,
                    split_pos[batch_i * minibatch:(batch_i + 1) * minibatch]
                )
            # Train the rest if it has
            if n_batches * minibatch != self.y.shape[0]:
                self.batch_train(
                    self.x[(batch_i + 1) * minibatch:],
                    self.y[(batch_i + 1) * minibatch:],
                    lr,
                    split_pos[(batch_i + 1) * minibatch:]
                )
            label_preds = self.predict(self.x, split_pos)
            error = metrics.zero_one_loss(self.label_y, label_preds)
            cost = self.cost(self.x, self.y, split_pos)
            if verbose:
                logging.info("epoch: %d training,on train data, "
                             "cross-entropy:%f, zero-one loss: %f"
                             % (epoch, cost, error))
            if abs(error - 0.0) <= 0.0001:
                break
        return epoch

    def predict(self, x, split_pos=None):
        """
        Prediction of FNN on x

        x: numpy.ndarray, 2d arry
            The input data. The index of words
        split_pos: 1d array like
            Start position in x. BRNN will compute from split_pos. The
            left_layer will compute from 0 to split_pos(not included)
            and the right_layer will compute from split_pos to last.
            If split_pos is None, split_pos will
            be the half of current row of x.
        Return
        -----
        numpy.ndarray, 1d array. The predict label on x
        """
        py = self.forward(x, split_pos)
        y = py.argmax(axis=1)
        return np.array([self.y_to_label[i] for i in y])


def brnn_test():
    x_col = 10
    no_softmax = 5
    n_h = 30
    up_wordvec = False
    use_bias = True
    act_func = 'tanh'
    use_lstm = True
    x_row = 100
    voc_size = 20
    word_dim = 4
    x = np.random.randint(low=0, high=voc_size, size=(x_row, x_col))
    # x = make_jagged_array(n_row=x_row, min_col=2, max_col=5,
                          # max_int=voc_size, min_int=0, dim_unit=None)
    label_y = np.random.randint(low=0, high=20, size=x_row)
    word2vec = np.random.uniform(low=0, high=5, size=(voc_size, word_dim))
    nntest = BRNN(x, label_y, word2vec, n_h, up_wordvec, use_bias,
                 act_func, use_lstm=use_lstm)
    split_pos = np.random.randint(low=4, high=8, size=(x_row, ))

    # Training
    lr = 0.1
    minibatch = 5
    max_epochs = 100
    verbose = True
    nntest.minibatch_train(lr, minibatch, max_epochs, split_pos, verbose)


def brnn_gradient_test():
    x_col = 3
    no_softmax = 5
    n_h = 2
    up_wordvec = False
    use_bias = True
    act_func = 'tanh'
    use_lstm = True
    x_row = 4
    voc_size = 20
    word_dim = 2
    # x = np.random.randint(low=0, high=voc_size, size=(x_row, x_col))
    x = make_jagged_array(n_row=x_row, min_col=2, max_col=5,
                          max_int=voc_size, min_int=0, dim_unit=None)
    label_y = np.random.randint(low=0, high=20, size=x_row)
    word2vec = np.random.uniform(low=0, high=5, size=(voc_size, word_dim))
    nntest = BRNN(x, label_y, word2vec, n_h, up_wordvec, use_bias,
                 act_func, use_lstm=use_lstm)

    # Gradient testing
    y = np.array([nntest.label_to_y[i] for i in label_y])
    gc = GradientChecker(epsilon=1e-05)
    gc.check_nn(nntest, x, y)


if __name__ == "__main__":
    # brnn_test()
    brnn_gradient_test()
