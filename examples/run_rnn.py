#! /usr/bin/env python3
"""
Authors: fengyukun
Date:   2016/03/29
Brief:  Examples of running models
"""

import sys
import os
sys.path.append("../lib/")
sys.path.append("../utils/")
sys.path.append("../models/")
from inc import*
from tools import*
from data_loader import DataLoader
from metrics import*
from rnn import RNN
from collections import OrderedDict


def gen_print_info(field_names, values):
    """
    Generate print infomation
    field_names: 1d array-like, each element of field_names is string
    values: 1d array-like. The value of field_names
    return: string
    """
    if len(field_names) != len(values):
        logging.error("The length is not the same.field_names:%s, values:%s"
                      % (str(field_names), str(values)))
        raise Exception
    res = ""
    for i in range(0, len(field_names)):
        res += "%s:%s\t" % (field_names[i], values[i])
    return res


def run_fnn():
    p = OrderedDict([
        ("\nParameters for word vectors", ""),
        #  ("word2vec_path", "../data/sample_word2vec.txt"),
        ("word2vec_path", "../../word2vec/vector_model/glove.6B.300d.txt"),
        ("norm_vec", False),
        ("oov", "O_O_V"),
        ("\nParameters for loading data", ""),
        #  ("data_path", "../data/sample"),
        ("data_path", "../data/salsa-parsed"),
        ("left_win", -1),
        ("right_win", -1),
        ("use_verb", True),
        ("lower", True),
        ("use_padding", False),
        # Validation part and train_part are from train_data_path
        ("train_part", 0.7),
        ("test_part", 0.2),
        ("validation_part", 0.1),
        # Minimum number of sentences of training data
        ("minimum_sent_num", 70), # ATTENTION TO THIS
        # Minimum frame of verb of training data
        ("minimum_frame", 2), # ATTENTION TO THIS
        ("\nParameters for rnn model", ""),
        ("n_h", 45), # ATTENTION TO THIS
        ("up_wordvec", False),
        ("use_bias", True),
        ("act_func", "tanh"),
        ("use_lstm", True),
        ("max_epochs", 100),
        ("minibatch", 5),
        ("lr", 0.1),
        ("random_vectors", True), # ATTENTION TO THIS
        ("\nOther parameters", ""),
        ("training_detail", False), # ATTENTION TO THIS
        ("prediction_results", "../result/attention_results")
    ])
    result_file = "lstm_win%s_n_h%s_lr%s_%s" % (p["left_win"],
    p["n_h"], p["lr"], os.path.basename(p["data_path"]))

    if not os.path.isdir(p["prediction_results"]):
        os.system("mkdir -p %s" % p["prediction_results"])
    p["prediction_results"] += "/" + result_file

    #  if os.path.exists(p["prediction_results"]):
        #  print("%s has existed, reindicate a result file" %
              #  p["prediction_results"])
        #  exit(0)

    if p["random_vectors"]:
        vocab, invocab, word2vec = build_vocab(
            corpus_dir=p["data_path"], oov=p["oov"],
            random_wordvec=True, dimension=300
        )
    else:
        # Get vocabulary and word vectors
        vocab, invocab, word2vec = get_vocab_and_vectors(
            p["word2vec_path"], norm_only=p["norm_vec"], oov=p["oov"],
            oov_vec_padding=0., dtype=FLOAT, file_format="auto"
        )
    # Updating word vectors only happens for one verb
    #   So when one verb is done, word vectors should recover
    if p["up_wordvec"]:
        word2vec_bak = np.array(word2vec, copy=True)

    # Get data
    train_loader = DataLoader(
        data_path=p["data_path"], vocab=vocab, oov=p["oov"],
        left_win=p["left_win"], right_win=p["right_win"],
        use_verb=p["use_verb"], lower=p["lower"], use_padding=p["use_padding"]
    )
    train, test, validation = train_loader.get_data(
        p["train_part"], p["test_part"], p["validation_part"],
        sent_num_threshold=p["minimum_sent_num"],
        frame_threshold=p["minimum_frame"]
    )

    field_names = [
        'precision', 'recall', 'f-score',
        "sentence number (train data)",
        "sentence number (test data)",
        "frame number(test data)",
        "epoch (train process)",
        "sentence number (validation data)",
        "valid_fscore"
    ]

    # Average statistics over all verbs
    scores_overall = np.zeros(len(field_names), dtype=FLOAT)
    verb_counter = 0
    fh_pr = open(p["prediction_results"], "w")
    verbs = train.keys()
    for verb in verbs:
        verb_counter += 1
        # Recover the word vectors
        if p["up_wordvec"] and verb_counter != 1:
            word2vec = np.array(word2vec_bak, copy=True)
        # Build RNN model for each verb
        rnn = RNN(
            x=train[verb][0], label_y=train[verb][1],
            word2vec=word2vec, n_h=p["n_h"],
            up_wordvec=p["up_wordvec"], use_bias=p["use_bias"],
            act_func=p["act_func"], use_lstm=p["use_lstm"]

        )

        epoch = rnn.minibatch_train(
            lr=p["lr"],
            minibatch=p["minibatch"],
            max_epochs=p["max_epochs"],
            verbose=p["training_detail"]
        )

        y_pred = rnn.predict(test[verb][0])
        precision, recall, f_score, _, _ = standard_score(
            y_true=test[verb][1], y_pred=y_pred
        )
        valid_pred = rnn.predict(validation[verb][0])
        _, _, valid_f, _, _ = standard_score(
            y_true=validation[verb][1], y_pred=valid_pred
        )

        scores = [
            precision, recall, f_score,
            len(train[verb][1]),
            len(test[verb][1]),
            len(set(test[verb][1])),
            epoch,
            len(validation[verb][1]),
            valid_f
        ]
        scores_overall += scores
        print("current verb:%s, scores are:" % verb)
        print(gen_print_info(field_names, scores))
        print("current completeness:%d/%d, average scores over %d verbs are:"
              % (verb_counter, len(verbs), verb_counter))
        print(gen_print_info(field_names, scores_overall / verb_counter))

        # Print prediction results
        sents = indexs2sents(test[verb][0], invocab)
        print("verb: %s\tf-score:%f" % (verb, f_score), file=fh_pr)
        for i in range(0, len(test[verb][1])):
            is_true = True if test[verb][1][i] == y_pred[i] else False
            print("%s\tpredict:%s\ttrue:%s\t%s"
                  % (is_true, y_pred[i], test[verb][1][i],
                     " ".join(sents[i])), file=fh_pr)

    # File handles
    fhs = [fh_pr, sys.stdout]
    for fh in fhs:
        print(gen_params_info(p), file=fh)
        print("End of training and testing, the average "
              "infomation over %d verbs are:" % len(verbs), file=fh)
        print(gen_print_info(field_names, scores_overall / len(verbs)),
              file=fh)
    fh_pr.close()

if __name__ == "__main__":
    run_fnn()
