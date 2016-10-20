#! /usr/bin/env python3
"""
Authors: fengyukun
Date:  2016-07-08
Brief:  The implementation of baselines: Random and most frequent (MF)
"""

# Activate automatic float divison for python2.
from __future__ import division
import sys
sys.path.append("../lib/")
sys.path.append("../utils/")
from inc import*

class RandomSelector(object):
    """
    The random baseline selects uniformly and independently one of the 
    possible labels for one instance.
    """
    def __init__(self, labels):
        """
        labels: 1d array-like
            The labels
        """

        self.labels = labels
        self.uniq_labels = np.unique(labels) 

    def select_one(self, from_set=True):
        """
        Select one label from labels or the set of labels randomly 
        from_set: boolean
            True: select one label from the set of labels
            False: select one label from labels
        Return
        ----
        One of the labels
        """

        if from_set:
           random_index = np.random.randint(0, self.uniq_labels.shape[0])
           return self.uniq_labels[random_index]
        else:
            random_index = np.random.randint(0, len(self.labels))
            return self.labels[random_index]

    def select_array(self, length, from_set=True):
        """
        Select array from labels or the set of labels randomly 
        from_set: boolean
            True: select array from the set of labels
            False: select array from labels
        Return
        ----
        Array of the labels
        """

        res_array = [] 
        for i in range(0, length):
            res_array.append(self.select_one(from_set))
        return res_array

    def set_labels(labels):
        """
        Set labels
        labels: 1d array-like
            The labels
        """

        self.labels = labels
        self.uniq_labels = np.unique(labels) 


class MostFrequent(object):
    """
    The most frequent (MF) baseline selects the most frequent label for 
    one instance.
    """

    def __init__(self, labels):
        """
        labels: 1d array-like
            The labels
        """

        self.labels = labels
        counter = {}
        for label in self.labels:
            if label not in counter:
                counter[label] = 0
            counter[label] += 1
        most_freq = 0
        most_freq_label = None
        for label, freq in counter.items():
            if freq > most_freq:
                most_freq = freq
                most_freq_label = label
        self.most_freq_label = most_freq_label
    
    def select(self, length):
        """
        Select label which is most frequent in labels 
        length: int
            The length of the return array
        """
        return [self.most_freq_label for i in range(0, length)]


def random_selector_test():
    np.random.seed(10)
    from_set = True
    labels = ["a", "b", "a", "t", "c", "a", "a"]
    rs = RandomSelector(labels)
    print(rs.select_one(from_set))
    print(rs.select_array(10, from_set))


def most_freq_test():
    labels = ["a", "b", "a", "t", "c", "a", "a"]
    mf = MostFrequent(labels)
    print(mf.select(5))
    

if __name__ == "__main__":
    # random_selector_test()
    most_freq_test()
