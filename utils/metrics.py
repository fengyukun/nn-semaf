#! /usr/bin/env python3
"""
Authors: fengyukun
Date:   2016/03/28
Brief:  Metrics implementation
"""

from sklearn.metrics import precision_recall_fscore_support
import logging
logging.basicConfig(
    level=logging.DEBUG, 
    format=" [%(levelname)s]%(filename)s:%(lineno)s[function:%(funcName)s] %(message)s"
)

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

def standard_score(y_true, y_pred):
    """
    Calculating standard micro-average score for classification task

    y_true: 1d array-like, the true values
    y_pred: 1d array-like, the predicted values by a model

    return: [Precision(float), Recall(float), F-score(float)
            , class_id2fscore(dict), prop_class(dict)]
    """
    # Compute micro-average score(precision, recall, f-score)
    prec_overall, rec_overall, f_overall, _ = precision_recall_fscore_support(
        y_true, y_pred, beta=1.0, pos_label=None, average="micro"
    )
    # Compute score and proportion of occurrences for each class in y_true for further analysis
    class_ids = list(set(y_true))
    _, _, fscore_class, num_class = precision_recall_fscore_support(
        y_true, y_pred, beta=1.0, pos_label=None, average=None, labels=class_ids
    )
    class_id2fscore = dict(zip(class_ids, fscore_class))
    prop_class = [num / float(len(y_true)) for num in num_class]
    class_id2prop = dict(zip(class_ids, prop_class))


    return [prec_overall, rec_overall, f_overall, class_id2fscore, class_id2prop]

def test_bcubed_score():
    y_true = [1, 2, 4, 2, 1, 5, 3, 1, 5]
    y_pred = [1, 2, 3, 2, 1, 6, 3, 4, 10]
    res = bcubed_score(y_true, y_pred, level="class")
    print(res)

def test_standard_score():
    y_true = [0, 1, 0]
    y_pred = [0, 0, 0]
    res = standard_score(y_true, y_pred)
    print(res)

def test_zero_one_loss():
    y_true = [1, 1, 1]
    y_pred = [0, 1, 0]
    res = zero_one_loss(y_true, y_pred)
    print(res)


if __name__ == "__main__":
    # test_bcubed_score()
    # test_standard_score()
    test_zero_one_loss()

