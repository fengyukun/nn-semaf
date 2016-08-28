#! /usr/bin/env python3
"""
Authors: fengyukun
Date:   2016/03/29
Brief:  Examples of running models
"""

import sys
sys.path.append("../models/lib/")
sys.path.append("../utils/")
sys.path.append("../models/")
from inc import*
from tools import*
from data_loader import DataLoader
from metrics import*
from brnn import BRNN
from attention_birnn import ABiRNN
from collections import OrderedDict
from parameter_setting import*


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
    p = p_bilstm
    p["left_win"] = -1 
    p["right_win"] = -1
    p["lr"] = 0.1
    p["n_h"] = 55
    p["data_path"] = "../data/semeval_mic_train_and_test_with_extraction"
    p["minimum_sent_num"] = 0
    p["prediction_results"] = "../result/brnn_results/bilstm_semeval_gold.correct"
    p["minimum_frame"] = 0
    p["train_part"] = 0.7
    p["test_part"] = 0.3
    p["validation_part"] = 0.0
    p["norm_func"] = 'softmax'
    on_validation = False
    training_detail = False
    # Get vocabulary and word vectors
    vocab, invocab, word2vec = get_vocab_and_vectors(
        p["word2vec_path"], norm_only=p["norm_vec"], oov=p["oov"],
        oov_vec_padding=0., dtype=FLOAT, file_format="auto"
    )
    if p["random_vectors"]:
        word2vec = np.array(
            np.random.uniform(low=-0.5, high=0.5, size=word2vec.shape)
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
        frame_threshold=p["minimum_frame"], 
        verb_index=p["verb_index"]
    )
    if on_validation:
        test = validation

    field_names = [
        'precision', 'recall', 'f-score',
        "sentence number (train data)",
        "sentence number (test data)",
        "frame number(test data)",
        'epoch'
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
        # Build BRNN model for each verb
        rnn = BRNN(
            x=train[verb][0], label_y=train[verb][1],
            word2vec=word2vec, n_h=p["n_h"],
            up_wordvec=p["up_wordvec"], use_bias=p["use_bias"],
            act_func=p["act_func"], use_lstm=p["use_lstm"]
        )

        epoch = rnn.minibatch_train(
            lr=p["lr"],
            minibatch=p["minibatch"],
            max_epochs=p["max_epochs"],
            split_pos=train[verb][2],
            verbose=training_detail
        )

        y_pred = rnn.predict(test[verb][0], split_pos=test[verb][2])

        precision, recall, f_score, _, _ = standard_score(
            y_true=test[verb][1], y_pred=y_pred
        )
        scores = [
            precision, recall, f_score,
            len(train[verb][1]),
            len(test[verb][1]),
            len(set(test[verb][1])),
            epoch
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
            out_line = "%s\tpredict:%s\ttrue:%s\t" % (is_true, y_pred[i], test[verb][1][i])
            out_line += " ".join(sents[i])
            print(out_line, file=fh_pr)

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
