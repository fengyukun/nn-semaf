#! /usr/bin/env python3
"""
Authors: fengyukun
Date:   2016/03/22
Brief:  Some commonly used functions in other modules
"""

# For python2
from __future__ import print_function
import os
import string
import numpy as np
from random import shuffle
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s]%(filename)s:%(lineno)s[function:%(funcName)s] %(message)s"
)

# Global data type
FLOAT = 'float64'
INT = "int32"

def remove_punctuations(strg):
    """
    Remove punctuations in string
    strg: string type
    """
    strg = "".join(ch for ch in strg if ch not in string.punctuation)
    return strg

def gen_params_info(parameters):
    """ Generate paratmeters infomation
    param: parameters: dict type, containing parameters
    return string
    """
    res = "\n*****Parameters******\n"
    for param_name, val in parameters.items():
        res += "%s\t:\t%s\n" % (str(param_name), str(val))
    res += "\n"
    return res

def sents_trimming(sents):
    """
    trimming on each word in sents
    sents: list, sentece of sentences, each sentence is a list
    """
    for sent in sents:
        for i in range(0, len(sent)):
            sent[i] = sent[i].strip()

def sents2indexs(sents, vocab, oov):
    """
    sentences to word indexs
    sents: list, sentece of sentences, each sentence is a list
    vocab: dict, word to word index
    oov: string, out of vocabulary. Words not in vocab will be replaced with oov
    return list of list, each element is a int number
    """
    sents_indexs = []
    for sent in sents:
        sent_indexs = []
        for word in sent:
            if word not in vocab:
                word = oov
            word_index = vocab[word]
            sent_indexs.append(word_index)
        sents_indexs.append(sent_indexs)
    return sents_indexs

def indexs2sents(sents_indexs, index2word_vocab):
    """
    Convert sentences in word indexs to words.

    sents_indexs: sentences, a list of list where each element is a word index
    index2word_vocab: index to word
    return: list of list

    """
    sents = []
    for sent_indexs in sents_indexs:
        sent = []
        for word_index in sent_indexs:
            word = index2word_vocab[word_index]
            sent.append(word)
        sents.append(sent)
    return sents

def sent_indexs_trunc(sent, window, direction, padding_index,
                      use_padding=True):
    """
    Sentence indexs truncation, padding padding_index where necessary

    sent: sentence, a list where each element is a word index
    window: int, the length of truncation window.
        If window==-1, truncation will include all words in sentence
    direction: two option values. If direction=="left", truncation will begin
        from right to left. Otherwise it will begin from left to right
    padding_index: int, padding padding_index where necessary
    use_padding: bool, whether padding padding_index when window is bigger than
                sent
    return: a list
    """
    if window == -1:
        return sent + []

    if window > len(sent):
        if not use_padding:
            return sent + []
        else:
            padding = [padding_index for i in range(0, window - len(sent))]
            if direction == "left":
                return padding + sent
            else:
                return sent + padding
    else:
        if direction == "left":
            return sent[len(sent) - window:] + []
        else:
            return sent[0:window] + []

def shuffle_two_lists(lista, listb):
    """
    Shuffling two corresponding list with same length
    lista, listb: list type, two corresponding lists
    return shuffled list
    """

    if len(lista) != len(listb):
        logging.error("The two lists must be the same length: lista:%d\tlistb:%d"\
                % (len(lista), len(listb)))
        raise Exception

    lista_shuf = []
    listb_shuf = []
    indexs_shuf = [i for i in range(0, len(lista))]
    shuffle(indexs_shuf)
    for i in indexs_shuf:
        lista_shuf.append(lista[i])
        listb_shuf.append(listb[i])
    return (lista_shuf, listb_shuf)

def load_word_vectors(vector_path, add_oov=False, oov="O_O_V"):
    """Load GloVe vectors. The format of vector is one word and its float values per line seperated
    by only space characters (e.g., '\t', ' ').

    :vector_path: str, the GloVe word vectors path
    :add_oov: boolean, whether out-of-vocabulary word is added to word2vec, which is often helpful
    when meating some unknown words.
    :oov: str, if add_oov is true the defined oov will be added to word2vec; zero vectors will be
    used as the vectors of oov.
    of oov 
    return [vocab(dict), inverse_vocab(dict), word2vec(numpy.ndarray)]

    """

    #  First read to count the number of words as well as the dimension of each vector.
    logging.info("Start loading word vectors: %s", vector_path)
    first_line = True
    word_count = 0
    vector_dimension = 0
    fh = open(vector_path, "r")
    for word_line in fh:
        word_line = word_line.strip()
        if word_line == '':
            continue
        word_count += 1
        if first_line:
            try:
                word_vector = word_line.split()
                vector_dimension = len(word_vector) - 1
            except Exception as e:
                logging.error("Vector format error.")
                raise Exception
            first_line = False
    fh.close()

    # Second read to allocate the whole vectors based on the first read.
    if add_oov:
        word_count += 1
    word2vec = np.zeros(shape=(word_count, vector_dimension), dtype=float)
    vocab = {}
    invocab = {}
    word_index = 0
    line_index = 0
    fh = open(vector_path, "r")
    for word_line in fh:
        line_index += 1
        word_line = word_line.strip()
        if word_line == '':
            continue
        try:
            word_vector = word_line.split()
            word = word_vector[0]
            vectors = [float(value) for value in word_vector[1:]]
            if len(vectors) != vector_dimension:
                raise Exception
        except Exception as e:
            logging.error("Vector format error. line: %s" % line_index)
            raise Exception

        for column_index in range(0, vector_dimension):
            word2vec[word_index][column_index] = vectors[column_index]
        vocab[word] = word_index
        invocab[word_index] = word
        word_index += 1
    fh.close()
    if add_oov:
        vocab[oov] = word_index
        invocab[word_index] = oov
    logging.info("Finish loading. Add oov: %s. Word vector (shape): %s"
                 % (add_oov, str(word2vec.shape)))

    return [vocab, invocab, word2vec]

def write_vocab_to_file(file_path, vocab, oov_tag=None):
    """Write vocabularies to a given file

    :file_path: str, the file path to write
    :vocab: dict, key is word and value is index.
    :oov_tag: str or None, out-of-vocabulary tag.

    """

    out_file = open(file_path, "w")
    print(str(oov_tag), file=out_file)
    for word, index in vocab.items():
        print("%s,%s" % (word, index), file=out_file)
    out_file.close()

def load_vocab_from_file(file_path):
    """Load vocabularies from a given file

    :file_path: str, the file path to load
    :returns: [vocab, oov_tag]. The key of dict is word and value of dict is index.

    """
    load_file = open(file_path, "r")
    vocab = {}
    oov_tag = load_file.readline().strip()
    if oov_tag == 'None':
        oov_tag = None
    for line in load_file:
        line = line.strip()
        if line == '':
            continue
        try:
            word, index = line.split(",")
        except:
            raise Exception("%s format error" % file_path)
        vocab[word] = index
    load_file.close()
    return [vocab, oov_tag]

def build_vocab(corpus_dir, oov="O_O_V", random_wordvec=False, dimension=300):
    """Build vocabulary from given corpus

    :corpus_dir: The directory of the given corpus. Multiple files will be
    count in corpus_dir.
    :oov: Out of vocabulary. Defining a oov is helpful when meet some unknown
    words in the test dataset
    :random_wordvec: Whether generate random word vector
    :dimension: If random_wordvec is true, dimension indicates the dimension of
    the word vector.
    :returns: [vocab, invocab, word2vec(if random_wordvec is true)]
        vocab: a dict, key is word id and the value is word iteself
        invocab: a dict, key is word and the value is word id
        word2vec: numpy.ndarray, 2d, row index is the word id

    """

    # Dict, key is word id(Start from 0)
    vocab = {}
    invocab = {}
    word_counter = 0
    file_list = os.listdir(corpus_dir)
    for file_name in file_list:
        # Skip hidden file
        if file_name.find(".") == 0:
            continue
        file_path = "%s/%s" % (corpus_dir, file_name)
        fh = open(file_path, "r")
        for line in fh:
            line = line.strip()
            # Skip empty line
            if line == '':
                continue
            try:
                tokens = line.split()
            except:
                tokens = [line]
            for token in tokens:
                if token not in vocab:
                    vocab[token] = word_counter
                    invocab[word_counter] = token
                    word_counter += 1
        fh.close()

    # Add oov to vocab
    if oov not in vocab:
        vocab[oov] = max(vocab.values()) + 1
        invocab[vocab[oov]] = oov
    if random_wordvec:
        word2vec = np.random.uniform(
            low=-1.0, high=1.0, size=(len(vocab.keys()), dimension)
        )
        return [vocab, invocab, word2vec]
    return [vocab, invocab]

def basic_test():
    sents = [["I", "have ", "done"], ["apple", "and", "food"]]
    sents_trimming(sents)
    vocab = {"I":1, "have":2, "done":3, "apple":4, "oov":5}
    sents_indexs = sents2indexs(sents, vocab, "oov")
    index2word_vocab = {k:v for v, k in vocab.items()}
    print(sents)
    print(sents_indexs)
    print(indexs2sents(sents_indexs, index2word_vocab))
    sent = [i for i in range(0, 5)]
    truncated_sent = sent_indexs_trunc(sent, 7, "right", -1)
    # print(sent)
    # print(truncated_sent)
    lista = [1, 2, 3, 4, 5]
    listb = [1, 2, 3, 4, 5]
    lista, listb = shuffle_two_lists(lista, listb)
    # print(lista)
    # print(listb)

def sigmoid(x):
    """
    The sigmoid function. This is numerically-stable version
    x: float
    """
    if x >= 0:
        return 1 / (1 + np.exp(-x))
    else:
        exp_x = np.exp(x)
        return exp_x / (exp_x + 1)

def sigmoid_array(x):
    """
    Numerically-stable sigmoid function
    x: ndarray (float)
    """
    vfunc = np.vectorize(sigmoid)
    return vfunc(x)

def sigmoid_test():
    x = np.array([[1, 0, 0, 1000], [-1000, 0, 29, 0.32]])
    print(sigmoid_array(x))

def load_word_vectors_stdin_test():
    vector_path = "../data/word_vectors/glove.6B.300d.txt"
    vector_path = "../data/sample_word2vec.txt"
    vocab, invocab, word2vec = load_word_vectors(vector_path)
    print("word2vec shape: %s" % str(word2vec.shape))

    import fileinput
    for test_word in fileinput.input():
        test_word = test_word.strip()
        if test_word == '':
            continue
        if test_word not in vocab:
            print("%s not in %s" % (test_word, vector_path))
            continue
        print("the word vector of the test word:")
        print(word2vec[vocab[test_word]])

def load_word_vectors_test():
    vector_path = "../data/sample_word2vec.txt"
    vocab, invocab, word2vec = load_word_vectors(vector_path, add_oov=True, oov="O_O_V")
    print("word2vec shape: %s" % str(word2vec.shape))
    print("vocab:")
    print(vocab)
    print("\ninvocab:")
    print(invocab)
    print("\nword vectors:")
    print(word2vec)

def build_vocab_test(): 
    vocab, invocab, word2vec = build_vocab("../data/pdev", "O_O_V", True, 5)
    #  print(len(vocab.keys()))
    #  print(vocab)
    #  print(invocab)
    #  print(word2vec.shape)

if __name__ == "__main__":
    # basic_test()
    #  sigmoid_test()
    #  build_vocab_test()
    load_word_vectors_test()
    #  load_word_vectors_stdin_test()


