#! /usr/bin/env python3
"""
Authors: fengyukun
Date:   2016/03/28
Brief:  Metrics implementation
"""

# Activate automatic float divison for python2.
from __future__ import division
import logging
logging.basicConfig(
    level=logging.DEBUG, 
    format=" [%(levelname)s]%(filename)s:%(lineno)s[function:%(funcName)s] %(message)s"
)

def average_precision(y_trues, y_scores):
    """Average precision (AP) metric for ranking problem

    :y_trues: 1d array like. The value of y_trues is 0 or 1. 1 means hit.
    :y_scores: scores for each postion in y_trues
    :returns: float, the average precision score

    """
    # Ranking from highest to lowest according to y_scores
    # ranked_items is a list of 2-length tuples
    ranked_items = sorted(
        zip(y_scores, y_trues), key=lambda tup: tup[0], reverse=True
    )

    average_precision_score = 0
    # During the following loop, relative_item_num is equl to the correct
    # number of item.
    relative_item_num = 0
    for i in range(0, len(ranked_items)):
        is_hit = ranked_items[i][1]
        relative_item_num += is_hit
        average_precision_score += is_hit * (relative_item_num / (i + 1))
    # No relative item
    if relative_item_num == 0:
        return 1
    average_precision_score /= relative_item_num
    return average_precision_score
    
def mean_average_precision(y_trues_array, y_scores_array):
    """Mean Average Precision (MAP) score. If the length of two array is
    different, the score will be over the short part.

    :y_trues_array: 2d array-like (can be jagged)
    :y_scores_array: 2d array-like (can be jagged)
    :returns: float, MAP score

    """
    map_score = 0;
    number = min(len(y_trues_array), len(y_scores_array))
    for y_trues, y_scores in zip(y_trues_array, y_scores_array):
        map_score += average_precision(y_trues, y_scores)
    map_score /= number
    return map_score

def bcubed_score(y_true, y_pred, level="class"):
    """
    Calculating B-cubed Precision and Recall for clustering task.
        F-score is the harmonic mean of Precision and Recall. 

    y_true: 1d array-like, the true values
    y_pred: 1d array-like, the predicted values by a model
    level: string, the option values are instance and class. The B-cubed Precision
        will be computed on the level of each instance or each class. The default value
        is 'class'. Default option is also the offical metric in subtask 2 of SemEval-2015 task 15

    return: [Precision(float), Recall(float), F-score(float)]
    """
    precision = bcubed_precision(y_true, y_pred, level)
    recall = bcubed_precision(y_pred, y_true, level)
    f_score = 0.0
    if precision + recall > 0:
        f_score = (2 * precision * recall) / float(precision + recall)
    return [precision, recall, f_score]

def bcubed_precision(y_true, y_pred, level="class"):
    """
    Calculating B-cubed Precision for clustering task.

    Class level:
    For each class i in y_pred:
        B-cubed_Precision_i = pairs_i found in y_true / pairs_i, 
    where pairs_i is the number of arbitrary combination of two
    instance (include itself) in class i

    y_true: 1d array-like, the true values
    y_pred: 1d array-like, the predicted values by a model
    level: string, the option values are instance and class. The B-cubed Precision
        will be computed on the level of each instance or each class
    
    """
    if len(y_pred) == 0:
        return 0.0

    pred_cl2inst = {}   # class id to instance indexs
    for inst in range(0, len(y_pred)):
        class_id = y_pred[inst]
        if class_id not in pred_cl2inst:
            pred_cl2inst[class_id] = []
        pred_cl2inst[class_id].append(inst)

    precision_overall_class = 0
    precision_overall_inst = 0
    for class_id, insts in pred_cl2inst.items():
        correct_class = 0
        for inst_i in insts:
            correct_inst = 0
            for inst_j in insts:
                if inst_i < len(y_true) and inst_j < len(y_true)\
                        and y_true[inst_i] == y_true[inst_j]:
                    correct_class += 1
                    correct_inst += 1
            precision_inst = correct_inst / float(len(insts))
            precision_overall_inst += precision_inst
        precision_class = correct_class / float(len(insts) ** 2)
        precision_overall_class += precision_class
    average_precision_class = precision_overall_class / float(len(pred_cl2inst.keys()))
    average_precision_inst = precision_overall_inst / float(len(y_pred))
    return average_precision_class if level == "class" else average_precision_inst

def zero_one_loss(y_true, y_pred):
    """
    Calculating zero-one loss

    y_true: 1d array-like, the true values
    y_pred: 1d array-like, the predicted values by a model
    return the error rate
    """
    if len(y_true) != len(y_pred):
        logging.error("Not the same length")
        raise Exception

    error = 0
    for p, q in zip(y_true, y_pred):
        if p != q:
            error += 1
    return error / len(y_true)

def micro_average_f1(y_true, y_pred):
    """Calculate the micro-average presion, recall and F-score for classification tasks.

    :y_true: 1d array like, the true labels
    :y_pred: 1d array like, the predicted labels by a model.
    :returns: (presion, recall, fscore)

    """

    # Check the length of y_true and y_pred
    if len(y_true) != len(y_pred):
        logging.error("The length of y_true and y_pred is not the same")
        raise Exception

    # The set of labels is based on y_true.
    uniq_labels = list(set(y_true))
    # Build a dict inversing the uniq_labels for acceleration when accessing the position of
    # labels.
    label_to_pos = {uniq_labels[i]: i for i in range(0, len(uniq_labels))}

    # Build confusion matrix
    confusion_matrix = []

    # Init matrix (the row and column are considered as known labels and predicted labels
    # respectively)
    for i in range(0, len(uniq_labels)):
        confusion_matrix.append([0 for k in range(0, len(uniq_labels))])
    # Fill matrix
    for i in range(0, len(y_true)):
        if y_pred[i] not in uniq_labels:
            continue
        known_pos = label_to_pos[y_true[i]]
        predicted_pos = label_to_pos[y_pred[i]]
        confusion_matrix[known_pos][predicted_pos] += 1

    # Count the true positive (tp), false positive (fp) and false negative (fn) instances for each
    # label.
    global_tp = 0
    global_fp = 0
    global_fn = 0
    for i in range(0, len(uniq_labels)):
        true_positive = confusion_matrix[i][i]
        false_positive = 0
        for j in range(0, len(uniq_labels)):
            if i == j:
                continue
            false_positive += confusion_matrix[j][i]
        false_negtive = 0
        for j in range(0, len(uniq_labels)):
            if i == j:
                continue
            false_negtive += confusion_matrix[i][j]
        global_tp += true_positive
        global_fp += false_positive
        global_fn += false_negtive

    # Calculate the presion and recall globally. Note sometimes the model may not give any labels
    # in the y_true.
    if (global_tp + global_fp == 0):
        presion = 0
    else:
        presion = global_tp / (global_tp + global_fp)
    if (global_tp + global_fn == 0):
        recall = 0
    else:
        recall = global_tp / (global_tp + global_fn)

    # Calculate the F-score
    if (presion + recall == 0):
        fscore = 0
    else:
        fscore = 2 * presion * recall / (presion + recall)
    return (presion, recall, fscore)

def test_bcubed_score():
    y_true = [1, 2, 4, 2, 1, 5, 3, 1, 5]
    y_pred = [1, 2, 3, 2, 1, 6, 3, 4, 10]
    res = bcubed_score(y_true, y_pred, level="class")
    print(res)

def test_zero_one_loss():
    y_true = [1, 1, 1]
    y_pred = [0, 1, 0]
    res = zero_one_loss(y_true, y_pred)
    print(res)

def test_average_precision():
    y_trues1 = [1, 1, 0, 1, 0, 0, 0, 0, 0, 0]
    y_scores1 = [(10 - x) / 10 for x in range(0, 10)]
    print("result: %s" % average_precision(y_trues1, y_scores1))
    y_trues2 = [1, 0, 0, 1, 0, 0, 0, 1, 0, 0]
    y_scores2 = [(10 - x) / 10 for x in range(0, 10)]
    print("result: %s" % average_precision(y_trues2, y_scores2))
    y_trues3 = [0, 0, 1, 1]
    y_scores3 = [0.1, 0.4, 0.35, 0.8]
    print("result: %s" % average_precision(y_trues3, y_scores3))

    y_trues_array = [y_trues1, y_trues2, y_trues3]
    y_scores_array = [y_scores1, y_scores2, y_scores3]
    map_score = mean_average_precision(y_trues_array, y_scores_array)
    print("map result: %s" % map_score)

def test_micro_average_fscore():
    y_true = [2, 3, 4, 0, 1, 0]
    y_pred = [0, 0, 0, 4, 1, 0]
    #  y_true = [0, 1, 0]
    #  y_pred = [2, 3, 4]

    res = micro_average_f1(y_true, y_pred)
    print(res)



if __name__ == "__main__":
    # test_bcubed_score()
    #  test_zero_one_loss()
    #  test_average_precision()
    test_micro_average_fscore()

