#! /usr/bin/env python3
"""
Authors: fengyukun
Date: 2016-08-16
Brief:  The implementation of Attention-based Bidirectional Recurrent Neural Network (ABiRNN)
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
import birecurrent_layer
from layer import FuncNormLayer
from layer import AttentionLayer
import copy


class ABiRNN(object):
    """
    Attention-based Bidirectional Recurrent Neural Network (ABiRNN) class
    """
    def init(self, x, label_y, word2vec, n_h, up_wordvec=False,
             use_bias=True, act_func='tanh',
             use_lstm=True, norm_func='softmax', global_independent=False):
        """
        Init ABRiNN
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
            Whether use lstm layer, default is lstm layer
        norm_func: str
            Attention normalization function.
            Two options are 'softmax' and 'sigmoid'
        global_independent: bool
            True: the global infomation in x wil be independent and not compute
            together with x.
            False: the global infomation inx will be treated equally as all x.
            The position of global infomation is specified by split_pos
        """

        self.x = x
        self.word2vec = word2vec
        self.up_wordvec = up_wordvec
        self.n_h = n_h
        self.act_func = act_func
        self.use_bias = use_bias
        self.use_lstm = use_lstm
        self.norm_func = norm_func
        self.global_independent = global_independent

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
        self.bir_layer = birecurrent_layer.BiRecurrentLayer()
        self.bir_layer.init_layer(n_i=self.n_i, n_o=self.n_h,
                                  act_func=self.act_func,
                                  use_bias=self.use_bias,
                                  use_lstm=self.use_lstm)

        self.params += self.bir_layer.params
        self.param_names += self.bir_layer.param_names

        # Init attention layer
        self.attention_layer = AttentionLayer(norm_func=self.norm_func)

        # Output layer
        self.softmax_layer = layer.SoftmaxLayer()
        self.softmax_layer.init_layer(n_i=self.n_h, n_o=self.n_o,
                                 use_bias=self.use_bias)
        self.params += self.softmax_layer.params
        self.param_names += self.softmax_layer.param_names

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
        print("%s %s %s %s %s %s %s %s %s" % (self.n_i, self.n_o, self.act_func, self.use_bias,
              self.use_lstm, self.n_h, self.up_wordvec, self.norm_func, self.global_independent),
              file=attributes_file)
        y_to_label = ",".join(['%s:%s' % (k, v) for k, v in self.y_to_label.items()]) 
        print(y_to_label, file=attributes_file)
        attributes_file.close()

        # Write paramters to file
        self.embedding_layer.write_to_files("%s/embedding_out.npz" % (target_dir,))
        bilayer_dir = "%s/%s" % (target_dir, self.bir_layer.__class__.__name__)
        self.bir_layer.write_to_files(bilayer_dir)
        softmax_target_dir = "%s/%s" % (target_dir, self.softmax_layer.__class__.__name__)
        self.softmax_layer.write_to_files(softmax_target_dir)
        logging.info("Finish writting %s layer to %s" % (self.__class__.__name__, target_dir))

    def load_from_files(self, target_dir):
        """Load files to recover one object of this class.

        :target_dir: str, a directory where the attribute file and paramter file are.

        """

        # Load attributes file
        attributes_file = open("%s/attributes.txt" % target_dir, "r")
        try:
            (n_i, n_o, act_func, use_bias, use_lstm, n_h, up_wordvec,
             norm_func, global_independent) = (attributes_file.readline().strip().split(" "))
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
            self.norm_func = norm_func
            if global_independent == 'True':
                self.global_independent = True
            else:
                self.global_independent = False
            y_to_label = attributes_file.readline().strip().split(',')
            self.y_to_label = {}
            for key_value in y_to_label:
                key, value = key_value.split(":")
                self.y_to_label[int(key)] = value
        except:
            raise Exception("%s/attributes.txt format error" % target_dir)
        attributes_file.close()
        
        # Load parameters file

        self.embedding_layer = layer.EmbeddingLayer()
        self.embedding_layer.load_from_files("%s/embedding_out.npz" % (target_dir,))

        self.layers = []
        self.params = []
        self.param_names = []

        self.bir_layer = birecurrent_layer.BiRecurrentLayer()
        bilayer_dir = "%s/%s" % (target_dir, self.bir_layer.__class__.__name__)
        self.bir_layer.load_from_files(bilayer_dir)
        self.params += self.bir_layer.params
        self.param_names += self.bir_layer.param_names

        # Init attention layer
        self.attention_layer = AttentionLayer(norm_func=self.norm_func)

        self.softmax_layer = layer.SoftmaxLayer()
        softmax_target_dir = "%s/%s" % (target_dir, self.softmax_layer.__class__.__name__)
        self.softmax_layer.load_from_files(softmax_target_dir)
        self.params += self.softmax_layer.params
        self.param_names += self.softmax_layer.param_names

        logging.info("Finish loading %s from %s" % (self.__class__.__name__, target_dir))

    def cost(self, x, y, split_pos=None):
        """
        Cost function
        split_pos: 1d array like
            Start position in x. ABiRNN will compute from split_pos. The
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
            Start position in x. These positions are used to compute the global
            infomation. If it is None, half of each row in x will be used as
            default.
        """

        embedding_out = self.embedding_layer.forward(x, input_opt='jagged')
        birlayer_out = self.bir_layer.forward(embedding_out)

        # IF split_pos is None then half of each row is used
        if split_pos is None:
            split_pos = [int(len(row) / 2) for row in birlayer_out]

        global_info = get_col_from_jagged_array(split_pos, birlayer_out)
        weighted_sums, attention_matrix = (
            self.attention_layer.forward(birlayer_out, global_info)
        )

        self.attention_matrix = attention_matrix
        self.birlayer_out = birlayer_out
        self.forward_out = self.softmax_layer.forward(weighted_sums)
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

        gbirlayer_out, gglobal_info = self.attention_layer.backprop(go)
        # Add gglobal_info to gx
        for i in range(0, len(gbirlayer_out)):
            gbirlayer_out[i][self.split_pos[i]] += gglobal_info[i]

        gx = self.bir_layer.backprop(gbirlayer_out)

        self.gparams = self.bir_layer.gparams + self.gparams
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
            Start position in x. ABiRNN will compute from split_pos. The
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
                        split_pos=None, verbose=False,
                        training_method='dynamic', stable_method='zero_one_loss'):
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
            Start position in x. TRNN will compute from split_pos. The
            left_layer will compute from 0 to split_pos(not included)
            and the right_layer will compute from split_pos to last.
            If split_pos is None, split_pos will
            be the half of current row of x.
        verbose: bool
            whether to print information during each epoch training
        training_method: str, two options are:
            dynamic: The leaning rate is dynamically adjusted. 
            fixed: The learning rate is fixed.
        stable_method: two options are:
            'zero_one_loss': The training considers to be stable when zero one loss is zero.
            'cost_stable': The training considers to be stable when the cost is continuously
            stable. The 'fixed' and 'cost_table' combination are not supported.
        Return
        ----
        train_epoch: int
            The epoch number during traing on train data
        """

        if training_method not in ['dynamic', 'fixed']:
            logging.error("Unknown training method argument: %s" % training_method)
            raise Exception
        if stable_method not in ['zero_one_loss', 'cost_stable']:
            logging.error("Unknown stable method argument: %s" % training_method)
            raise Exception
        if stable_method == 'cost_stable' and training_method == 'fixed':
            logging.error("Current combination is not supported")
            raise Exception

        if training_method == 'dynamic':
            last_cost = None
            stable_threshold = 2 
            stable_max_times = 3
            stable_times = 0

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

            if training_method == 'dynamic':
                # The first epoch
                if last_cost is None:
                    last_cost = cost
                    continue
                # If the cost is stable for stable_max_times within stable_threshold,
                # the training is stopped.
                if stable_method == 'cost_stable':
                    if abs(cost - last_cost) <= stable_threshold:
                        stable_times += 1
                        if verbose:
                            logging.info("The cost is continuously stable for %s times" % stable_times)
                        if stable_times >= stable_max_times:
                            break
                    else:
                        stable_times = 0
                if stable_method == 'zero_one_loss':
                    if abs(error - 0.0) <= 0.0001:
                        break

                # Dynamically adjust the learning rate.
                # If cost is reduced bigger than reduced_percentage, the learning rate is increased
                # by increased_percentage.
                reduced_percentage = 0.10
                increased_percentage = 0.05
                diff = last_cost - cost
                if (diff > 0 and (diff / last_cost) >= reduced_percentage):
                    lr *= (1 + increased_percentage)
                    if verbose:
                        logging.info("The cost has been reduced by more than %s. Learning rate "\
                                "is increased to %s" % (reduced_percentage, lr))
                # If cost is actually increasing by cost_dec_percentage, decrease the learning rate
                # by decrease_percentage
                decrease_percentage = 0.05
                cost_dec_percentage = 0.05
                if diff < 0 and abs(diff) / last_cost >= cost_dec_percentage:
                    lr *= (1 - decrease_percentage) 
                    if verbose:
                        logging.info("The cost increased. Learning rate is decreased to %s" % lr)

                last_cost = cost

            if training_method == 'fixed' and stable_method == 'zero_one_loss':
                # If the zero-one loss is zero, the training is stopped.
                if abs(error - 0.0) <= 0.0001:
                    break

        return epoch

    def predict(self, x, split_pos=None):
        """
        Prediction of FNN on x

        x: numpy.ndarray, 2d arry
            The input data. The index of words
        split_pos: 1d array like
            Start position in x. ABiRNN will compute from split_pos. The
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
    nntest = ABiRNN()
    nntest.init(x, label_y, word2vec, n_h, up_wordvec, use_bias,
                act_func, use_lstm=use_lstm)
    split_pos = np.random.randint(low=4, high=8, size=(x_row, ))

    # Training
    lr = 0.1
    minibatch = 5
    max_epochs = 100
    verbose = True
    training_method = 'dynamic'
    #  training_method = 'fixed'
    stable_method = 'zero_one_loss'
    #  stable_method = 'cost_stable'
    nntest.minibatch_train(lr, minibatch, max_epochs, split_pos, verbose, training_method,
                           stable_method)


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
    nntest = ABiRNN()
    nntest.init(x, label_y, word2vec, n_h, up_wordvec, use_bias,
                act_func, use_lstm=use_lstm, norm_func='softmax',
                global_independent=True)

    # Gradient testing
    y = np.array([nntest.label_to_y[i] for i in label_y])
    gc = GradientChecker(epsilon=1e-05)
    gc.check_nn(nntest, x, y)

    # Write and load test
    nntest_bak = ABiRNN()
    nntest.write_to_files("abirnn_dir")
    nntest_bak.load_from_files("abirnn_dir")
    print("After loading")
    gc = GradientChecker(epsilon=1e-05)
    gc.check_nn(nntest_bak, x, y)
    print("Orignal output")
    print(nntest.forward(x))
    print("After loading output")
    print(nntest_bak.forward(x))

if __name__ == "__main__":
    brnn_test()
    #  brnn_gradient_test()
