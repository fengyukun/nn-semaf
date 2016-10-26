#! /usr/bin/env python3
"""
Authors: fengyukun
Date:   2016/03/29
Brief:  Examples of running models
"""

# For python2
from __future__ import print_function
# Activate automatic float divison for python2.
from __future__ import division
import sys
sys.path.append("../lib/")
sys.path.append("../utils/")
sys.path.append("../models/")
from inc import*
from tools import*
from data_loader import DataLoader
from metrics import*
from abirnn import ABiRNN
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

def train_and_save_model():
    """Train and save the model to the file for later prediction.

    """
    p = OrderedDict([
        ("\nParameters for word vectors", ""),
        #  ("word2vec_path", "../data/sample_word2vec.txt"),
        ("word2vec_path", "../../data/word_vectors/glove.6B.300d.txt"),
        ("oov", "O_O_V"),
        ("\nParameters for loading data", ""),
        ("train_path", "../../data/corpus/wsj_framnet/"),
        ("left_win", -1),
        ("right_win", -1),
        ("use_verb", True),
        ("lower", True),
        ("use_padding", False),
        ("verb_index", True),
        # Validation part and train_part are from train_data_path
        ("train_part", 1.0),
        ("test_part", 0.0),
        ("validation_part", 0.0),
        # Minimum number of sentences of training data
        ("minimum_sent_num", 0), # ATTENTION TO THIS
        # Minimum frame of verb of training data
        ("minimum_frame", 0), # ATTENTION TO THIS
        ("\nParameters for rnn model", ""),
        ("n_h", 100), # ATTENTION TO THIS
        ("up_wordvec", False),
        ("use_bias", True),
        ("act_func", "tanh"),
        ("use_lstm", True),
        ("max_epochs", 100),
        ("minibatch", 50), # ATTENTION TO THIS
        ("lr", 0.1),
        ("training_method", "fixed"),
        ("stable_method", "zero_one_loss"),
        ("norm_func",'softmax'),
        ("show_key_words", False), # ATTENTION TO THIS
        ("key_words_tag", "keywordtag"),
        ("random_vectors", False), # ATTENTION TO THIS
        ("\nOther parameters", ""),
        ("training_detail", True), # ATTENTION TO THIS
        ("result_dir", "../../results/nnfl/trained_models/wsjfnfull.abirnn.fixed.model"),
        ("vocab_path", "../../results/nnfl/trained_models/wsjfnfull.abirnn.fixed.model/vocab")
    ])

    # Get the word vectors
    if p["random_vectors"]:
        vocab, invocab, word2vec = build_vocab(
            corpus_dir=p["train_path"], oov=p["oov"],
            random_wordvec=True, dimension=300
        )
    else:
        # Get vocabulary and word vectors
        vocab, invocab, word2vec = load_word_vectors(
            p["word2vec_path"], add_oov=True, oov=p["oov"]
        )

    # Get train data
    train_loader = DataLoader(
        data_path=p["train_path"], vocab=vocab, oov=p["oov"],
        left_win=p["left_win"], right_win=p["right_win"],
        use_verb=p["use_verb"], lower=p["lower"], use_padding=p["use_padding"],
        show_key_words=p["show_key_words"], key_words_tag=p["key_words_tag"]
    )
    train, _, _ = train_loader.get_data(
        p["train_part"], p["test_part"], p["validation_part"],
        sent_num_threshold=0,
        frame_threshold=p["minimum_frame"], 
        verb_index=p["verb_index"]
    )

    train_file = train.keys()[0]

    # Train
    nn = ABiRNN()
    nn.init(
        x=train[train_file][0], label_y=train[train_file][1],
        word2vec=word2vec, n_h=p["n_h"],
        up_wordvec=p["up_wordvec"], use_bias=p["use_bias"],
        act_func=p["act_func"], use_lstm=p["use_lstm"],
        norm_func=p["norm_func"]
    )
    epoch = nn.minibatch_train(
        lr=p["lr"],
        minibatch=p["minibatch"],
        max_epochs=p["max_epochs"],
        split_pos=train[train_file][2],
        verbose=p["training_detail"],
        training_method=p["training_method"],
        stable_method=p["stable_method"]
    )

    # Write the model to file
    nn.write_to_files(p["result_dir"])
    # Write the vocab to file
    write_vocab_to_file(p["vocab_path"], vocab, oov_tag=p["oov"])

def load_and_test():
    model_path = "../../results/nnfl/trained_models/wsjfn_721.abirnn.model"
    p = OrderedDict([
        #  ("test_path", "../../data/corpus/semeval_mic_test_and_pdev_train/test/"),
        #  ("test_path", "../../data/corpus/semeval_wing_test"),
        ("test_path", "../../data/corpus/semeval_mic_train_and_test_with_key_words/"),
        ("left_win", -1),
        ("right_win", -1),
        ("use_verb", True),
        ("lower", True),
        ("use_padding", False),
        ("verb_index", True),
        ("show_key_words",True), # ATTENTION TO THIS
        ("key_words_tag", "keywordtag"),
        # Minimum number of sentences of training data
        ("minimum_sent_num", 0), # ATTENTION TO THIS
        # Minimum frame of verb of training data
        ("minimum_frame", 0), # ATTENTION TO THIS
        ("model_path", model_path),
        ("vocab_path", "%s/vocab" % model_path)
    ])

    # Load the vocab
    vocab, oov_tag = load_vocab_from_file(p["vocab_path"])
    # Get the test data
    test_loader = DataLoader(
        data_path=p["test_path"], vocab=vocab, oov=oov_tag,
        left_win=p["left_win"], right_win=p["right_win"],
        use_verb=p["use_verb"], lower=p["lower"], use_padding=p["use_padding"],
        show_key_words=p["show_key_words"], key_words_tag=p["key_words_tag"]
    )
    _, test, _ = test_loader.get_data(
        0.0, 1.0, 0.0,
        sent_num_threshold=0,
        frame_threshold=0, 
        verb_index=p["verb_index"]
    )

    # Load model
    nn = ABiRNN()
    nn.load_from_files(p["model_path"])

    field_names = [
        'precision', 'recall', 'f-score',
        "sentence number (test data)",
        "frame number(test data)",
    ]
    if p["show_key_words"]:
        field_names.append("test_map_score")

    # Average statistics over all verbs
    scores_overall = np.zeros(len(field_names), dtype=FLOAT)
    verb_counter = 0
    verbs = test.keys()
    for verb in verbs:
        verb_counter += 1
        y_pred = nn.predict(test[verb][0], split_pos=test[verb][2])
        precision, recall, f_score = bcubed_score(
            y_true=test[verb][1], y_pred=y_pred
        )
        attention_matrix = nn.attention_matrix
        test_p, test_r, test_f = micro_average_score(
            y_true=test[verb][1], y_pred=y_pred
        )

        scores = [
            precision, recall, f_score,
            len(test[verb][1]),
            len(set(test[verb][1])),
        ]
        if p["show_key_words"]:
            map_score = mean_average_precision(
                y_trues_array=test[verb][3], y_scores_array=attention_matrix
            )
            scores.append(map_score)

        scores_overall += scores
        print("current verb:%s, scores are:" % verb)
        print(gen_print_info(field_names, scores))
        print("current completeness:%d/%d, average scores over %d verbs are:"
              % (verb_counter, len(verbs), verb_counter))
        print(gen_print_info(field_names, scores_overall / verb_counter))

    print(gen_params_info(p))
    print("End of training and testing, the average "
          "infomation over %d verbs are:" % len(verbs))
    print(gen_print_info(field_names, scores_overall / len(verbs)))

def train_and_test():
    p = OrderedDict([
        ("\nParameters for word vectors", ""),
        ("word2vec_path", "../data/sample_word2vec.txt"),
        #  ("word2vec_path", "../../data/word_vectors/glove.6B.300d.txt"),
        ("oov", "O_O_V"),
        ("\nParameters for loading data", ""),
        #  ("data_path", "../data/sample"),
        ("train_path", "../../data/corpus/semeval_mic_test_and_pdev_train/train"),
        ("test_path", "../../data/corpus/semeval_mic_test_and_pdev_train/test"),
        ("left_win", -1),
        ("right_win", -1),
        ("use_verb", True),
        ("lower", True),
        ("use_padding", False),
        ("verb_index", True),
        # Minimum number of sentences of training data
        ("minimum_sent_num", 0), # ATTENTION TO THIS
        # Minimum frame of verb of training data
        ("minimum_frame", 0), # ATTENTION TO THIS
        ("\nParameters for rnn model", ""),
        ("n_h", 45), # ATTENTION TO THIS
        ("up_wordvec", False),
        ("use_bias", True),
        ("act_func", "tanh"),
        ("use_lstm", True),
        ("max_epochs", 1),
        ("minibatch", 10),
        ("lr", 0.1),
        ("training_method", "dynamic"),
        ("stable_method", "zero_one_loss"),
        ("norm_func",'softmax'),
        ("random_vectors", False), # ATTENTION TO THIS
        ("show_key_words",True), # ATTENTION TO THIS
        ("key_words_tag", "keywordtag"),
        ("\nOther parameters", ""),
        ("training_detail", False), # ATTENTION TO THIS
        ("prediction_results", "../../results/nnfl/abinn/abinn_test_logic"),
        # For SemEval-2007 task 06
        ("out_dir", "../../results/nnfl/brnn/semeval07task06_out_preprocessd_fixed")
    ])

    os.system("mkdir -p %s" % p["out_dir"])

    if p["random_vectors"]:
        vocab, invocab, word2vec = build_vocab(
            corpus_dir=p["train_path"], oov=p["oov"],
            random_wordvec=True, dimension=300
        )
    else:
        # Get vocabulary and word vectors
        vocab, invocab, word2vec = load_word_vectors(
            p["word2vec_path"], add_oov=True,oov=p["oov"]
        )

    # Get train data
    train_loader = DataLoader(
        data_path=p["train_path"], vocab=vocab, oov=p["oov"],
        left_win=p["left_win"], right_win=p["right_win"],
        use_verb=p["use_verb"], lower=p["lower"], use_padding=p["use_padding"],
        show_key_words=p["show_key_words"], key_words_tag=p["key_words_tag"]
    )
    train, _, validation = train_loader.get_data(
        1.0, 0.0, 0.0,
        sent_num_threshold=0,
        frame_threshold=p["minimum_frame"], 
        verb_index=p["verb_index"]
    )

    # Get the test data
    test_loader = DataLoader(
        data_path=p["test_path"], vocab=vocab, oov=p["oov"],
        left_win=p["left_win"], right_win=p["right_win"],
        use_verb=p["use_verb"], lower=p["lower"], use_padding=p["use_padding"],
        show_key_words=p["show_key_words"], key_words_tag=p["key_words_tag"]
    )
    _, test, _ = test_loader.get_data(
        0.0, 1.0, 0.0,
        sent_num_threshold=0,
        frame_threshold=0, 
        verb_index=p["verb_index"]
    )
    validation = test

    field_names = [
        'precision', 'recall', 'f-score',
        "sentence number (train data)",
        "sentence number (test data)",
        "frame number(test data)",
        "epoch (train process)",
        "sentence number (validation data)",
        "valid_fscore"
    ]
    if p["show_key_words"]:
        field_names.append("test_map_score")
        field_names.append("valid_map_score")

    # Average statistics over all verbs
    scores_overall = np.zeros(len(field_names), dtype=FLOAT)
    verb_counter = 0
    fh_pr = open(p["prediction_results"], "w")
    verbs = train.keys()
    for verb in verbs:
        verb_counter += 1
        # Build TRNN model for each verb
        rnn = ABiRNN()
        rnn.init(
            x=train[verb][0], label_y=train[verb][1],
            word2vec=word2vec, n_h=p["n_h"],
            up_wordvec=p["up_wordvec"], use_bias=p["use_bias"],
            act_func=p["act_func"], use_lstm=p["use_lstm"],
            norm_func=p["norm_func"]
        )

        epoch = rnn.minibatch_train(
            lr=p["lr"],
            minibatch=p["minibatch"],
            max_epochs=p["max_epochs"],
            split_pos=train[verb][2],
            verbose=p["training_detail"],
            training_method=p["training_method"],
            stable_method=p["stable_method"]
        )

        # Run trained model on test data
        y_pred = rnn.predict(test[verb][0], split_pos=test[verb][2])

        attention_matrix = rnn.attention_matrix
        test_p, test_r, test_f = micro_average_score(
            y_true=test[verb][1], y_pred=y_pred
        )

        # Output
        out_file = "%s/%s" % (p["out_dir"], verb)
        out_fh = open(out_file, "w")
        for instance_id, sense_tag in zip(test[verb][1], y_pred):
            print("%s %s %s" % (verb, instance_id, sense_tag), file=out_fh)
        out_fh.close()


        # Run trained model on validation data
        valid_pred = rnn.predict(
            validation[verb][0], split_pos=validation[verb][2]
        )
        valid_attention_matrix = rnn.attention_matrix
        valid_p, valid_r, valid_f = micro_average_score(
            y_true=validation[verb][1], y_pred=valid_pred
        )

        scores = [
            test_p, test_r, test_f,
            len(train[verb][1]),
            len(test[verb][1]),
            len(set(test[verb][1])),
            epoch,
            len(validation[verb][1]),
            valid_f
        ]
        if p["show_key_words"]:
            map_score = mean_average_precision(
                y_trues_array=test[verb][3], y_scores_array=attention_matrix
            )
            valid_map_score = mean_average_precision(
                y_trues_array=validation[verb][3],
                y_scores_array=valid_attention_matrix
            )
            scores.append(map_score)
            scores.append(valid_map_score)
        scores_overall += scores
        print("current verb:%s, scores are:" % verb)
        print(gen_print_info(field_names, scores))
        print("current completeness:%d/%d, average scores over %d verbs are:"
              % (verb_counter, len(verbs), verb_counter))
        print(gen_print_info(field_names, scores_overall / verb_counter))

        # Print prediction results
        sents = indexs2sents(test[verb][0], invocab)
        print("verb: %s\tf-score:%f" % (verb, test_f), file=fh_pr)
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
    train_and_save_model()
    #  load_and_test()
    #  train_and_test()
