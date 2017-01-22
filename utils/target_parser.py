#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: fengyukun
Date: 2016-11-18
Email: fengyukun_blcu@126.com
Github: https://github.com/fengyukun
Description: Target-specific parser implementated with Stanford dependency parser, NER and
WordNetLemmatizer of NLTK toolkit
"""

import os
import multiprocessing
import threading
from nltk.parse.stanford import StanfordDependencyParser
from nltk.tag import StanfordNERTagger
from nltk.stem.wordnet import WordNetLemmatizer

class TargetParser(object):
    """Target-specific parser. This class is designed to get the parsed result related to a
    given target in a sentence."""
    def __init__(self, dep_parser_jar_path, dep_parser_model_path,
                 ner_jar_path, ner_model_path):
        """Init the parser

        :dep_parser_jar_path: str, the jar path provided by Stanford dependency parser
        :dep_parser_model_path: str, the model path provided by Stanford dependency parser
        :ner_jar_path: str, the jar path provided by Stanford NER
        :ner_model_path: str, the model path provided by Stanford NER

        """
        self.dep_parser_jar_path = dep_parser_jar_path
        self.dep_parser_model_path = dep_parser_model_path
        self.ner_jar_path = ner_jar_path
        self.ner_model_path = ner_model_path
        self.dep_parser = StanfordDependencyParser(path_to_jar=self.dep_parser_jar_path,
                                                   path_to_models_jar=self.dep_parser_model_path)
        self.ner = StanfordNERTagger(model_filename=ner_model_path, path_to_jar=ner_jar_path)
        self.lmtzr = WordNetLemmatizer()
        
        # Get the number of CPUs on current machine
        self.cpu_count = multiprocessing.cpu_count()

    def parse_target(self, sent, target):
        """Parse the target in a sentence. All dependents related to the target will be extracted
        out as features as well as their part-of-speech (POS) tags, dependency relations, lemmas
        and named entities (NE). Since we did not find word index to locate the target in a
        sentence in the above tools, we simply search targets according to their names.  Targets
        which have the same forms in one sentence may cause some inaccuracies. 

        :sent: str, the sentence
        :target: str, target
        :retrun: list, the format is:
        [relation, depentent, depentent_ner, depentent_pos, depentent_lemma ... target, target_pos]
        :exception: raise Exception if error happens

        """

        parsed_sent = self.dep_parser.raw_parse(sent).next()
        ner_sent = self.ner.tag(sent.split())
        ner_dict = dict(ner_sent)
        result_list = []
        target_pos = ""
        for item in parsed_sent.triples():
            try:
                governor, governor_pos = item[0]
                relation = item[1]
                depentent, depentent_pos = item[2]
            except Exception as e:
                raise e
            if governor == target:
                target_pos = governor_pos
                # Get NE
                depentent_ner = "O"
                if depentent in ner_dict:
                    depentent_ner = ner_dict[depentent]
                # Get lemma
                depentent_lemma = depentent
                if depentent_pos.find("VB") >= 0:
                    depentent_lemma = self.lmtzr.lemmatize(depentent,'v')
                else:
                    depentent_lemma = self.lmtzr.lemmatize(depentent, 'n')
                result_list.extend(
                    [relation, depentent, depentent_ner, depentent_pos, depentent_lemma]
                )
        result_list.extend([target, target_pos])
        return result_list

    def parse_file(self, file_path, result_file, spliter="\t"):
        """Parse a given file with each sentence per line. The format is:
        [label][spliter][left_sentence][spliter][target][right_setence]

        :file_path: str
        :result_file: str
        :spliter:, str. The spliter in one sentence default as "\t"

        """

        fh_in = open(file_path, "r")
        fh_out = open(result_file, "w")
        for line in fh_in:
            line = line.strip("\n")
            if line == '':
                continue
            tokens = line.split(spliter)
            if len(tokens) != 4:
                print("%s spliter. Format error on line: %s, skip" % (len(tokens), line))
                continue
            label = tokens[0]
            target = tokens[2]
            sent = "%s %s %s" % (tokens[1], tokens[2], tokens[3])
            try:
                result_list = self.parse_target(sent, target)
            except Exception as e:
                print("Error happens during parsing %s. Skip it" % line)
                continue
            result_str = " ".join(result_list)
            fh_out.write(label + " " + result_str + "\n")
        fh_in.close()
        fh_out.close()

    def parse_in_dir(self, directory, verbose=True):
        """Parse in a given directory. Each non-hidden file in directory will be parsed into
        file.parsed

        :directory: str
        :verbose: bool

        """

        file_list = os.listdir(directory)
        for file_name in file_list:
            # Skip hidden file or parsed file (avoid reparsing)
            if file_name.find(".") == 0 or file_name.find(".parsed") >= 0:
                continue
            if verbose:
                print("To process %s" % file_name)
            file_path = "%s/%s" % (directory, file_name)
            parsed_file = "%s/%s.parsed" % (directory, file_name)
            self.parse_file(file_path, parsed_file)
        if verbose:
            print("Finished")

if __name__ == "__main__":
    # Stanford denpendency parser path
    dep_parser_path = "../../../softwares/stanford-corenlp-full-2014-08-27/"
    dep_parser_jar_path = dep_parser_path + "/stanford-corenlp-3.4.1.jar"
    dep_parser_model_path = dep_parser_path + "/stanford-corenlp-3.4.1-models.jar"
    # Stanford denpendency parser path
    ner_path = "../../../softwares/stanford-ner-2014-08-27/"
    ner_jar_path = ner_path + "/stanford-ner.jar"
    ner_model_path = ner_path + "/classifiers/english.all.3class.distsim.crf.ser.gz"

    # Inint TargetParser
    target_parser = TargetParser(
        dep_parser_jar_path=dep_parser_jar_path,
        dep_parser_model_path=dep_parser_model_path,
        ner_jar_path=ner_jar_path,
        ner_model_path=ner_model_path
    )
    target_path = "../../data/corpus/pdev_new_split/valid"
    #  target_path = "./test_target_parser/"
    target_parser.parse_in_dir(target_path)
