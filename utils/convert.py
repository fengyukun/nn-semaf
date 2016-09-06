#! /usr/bin/env python3
"""
Authors: fengyukun
Date:   2016/03/23
Brief:  Convert pdev data and SemEval 2015 data to the required format
"""

import os
import html
import nltk
from nltk.corpus import ptb
from nltk.corpus import propbank
from nltk.stem import WordNetLemmatizer
import codecs
import string
from tools import*
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format=" [%(levelname)s]%(filename)s:%(lineno)s[function:%(funcName)s] %(message)s"
)

def convert_chn_propbank(detail=True):
    """
    Convert Chinese propbank
    """
    chn_propbank_file = "../../../nltk_data/cpb2/data/verbs.txt.utf8"
    show_key_words = False
    key_word_tag = "keywordtag"

    #  out_dir = "../data/show_key_words_chn_propbank/"
    out_dir = "../data/chn_propbank/"
    os.system("rm -rf %s" % out_dir)
    os.system("mkdir -p %s" % out_dir)

    chn_punctuation = "》（）&%￥#@！{}【】，。；：、『「．"
    is_remove_punct = True

    ################################
    #  Load Chinese propbank file  #
    ################################

    # Verb to items (one item is real tagging information on Chinese treebank)
    verb_to_items = {}
    # Verb to its frames
    verb_to_frames = {}
    
    fh = open(chn_propbank_file)
    for instance in fh:
        instance = instance.strip()
        # Empty line
        if instance == "":
            continue
        items = instance.split()
        fileid = items[0]
        sent_id = items[1]
        verb_id = items[2]
        roleset = items[4]
        verb, frame_id = roleset.split(".")
        if verb not in verb_to_frames:
            verb_to_frames[verb] = []
        verb_to_frames[verb].append(frame_id)

        ###################################
        #  Find arguments of target verb  #
        ###################################
        # arguments, e.g., 0:1-ARG0=Agent 3:0-rel 6:2-ARG1=Topic
        arguments = items[6:]
        argument_list = []
        for argument in arguments:
            # position and tag label
            pos_labels = argument.split("-");
            pos = pos_labels[0]
            label = pos_labels[1]
            if label == "rel":
                continue
            pos = pos.replace(",", " ").replace("*", " ")
            # Single item, e.g., 22:1
            if pos.find(" ") < 0:
                items = [pos]
            else:
                items = pos.split(" ")
            # Multiple items
            for item in items:
                wordnum, height = item.split(":")
                wordnum = int(wordnum)
                height = int(height)
                pos_list = [
                    x for x in range(wordnum, wordnum + height + 1)
                ]
                argument_list.extend(pos_list)
        argument_list.append(int(verb_id))
        # Remove duplicate
        argument_list = list(set(argument_list))

        if verb not in verb_to_items:
            verb_to_items[verb] = []
        verb_to_items[verb].append(
            [fileid, sent_id, verb_id, frame_id, argument_list]
        )
    fh.close()

    ###############################################################
    #  Genrate data file from propbank file and Chinese treebank  #
    ###############################################################
    for verb, items in verb_to_items.items():
        # Skip those verbs which only have one frame in all instances
        if len(set(verb_to_frames[verb])) == 1:
            continue
        fh_out = open("%s/%s" % (out_dir, verb), "w")
        for item in items:
            fileid = item[0]
            sent_id = int(item[1])
            verb_id = int(item[2])
            frame_id = item[3]
            argument_list = item[4]

            fileid_for_ptb = "bracketed/%s" % fileid
            tagged_sent = ptb.tagged_sents(fileid_for_ptb)[sent_id]
            # Change tagged_sent from [tuples] to [list]
            tagged_sent = [[x[0], x[1]]for x in tagged_sent]

            # Show key words
            if show_key_words:
                for word_pos in range(0, len(tagged_sent)):
                    if word_pos not in argument_list:
                        continue
                    word = tagged_sent[word_pos][0]
                    if word.find("*") >=0 or word in chn_punctuation:
                        continue
                    tagged_sent[word_pos][0] += key_word_tag
            verb_bak = tagged_sent[verb_id][0]
            verb_identifier = "verb_identifier_xxxxx"
            tagged_sent[verb_id][0] = verb_identifier
            sent = []
            for (token, tag)in tagged_sent:
                if tag != '-NONE-':
                    if is_remove_punct and token in chn_punctuation:
                        continue
                    sent.append(token)
            sent = " ".join(sent)
            left_sent, right_sent = sent.split(verb_identifier)
            out_line = "%s\t%s\t%s\t%s" % (frame_id, left_sent, verb_bak,
                                           right_sent)
            print(out_line, file=fh_out)

        fh_out.close()  


def merge_split_data(detail=True):
    """
    Merge the split data.
    """

    merge_dirs = ["../data/split_semeval_mic_train_and_test_by_parser"]
    out_dirs = ["../data/merge_semeval_mic_train_and_test_by_parser"]

    for out_dir in out_dirs:
        os.system("rm -rf %s" % out_dir)
        os.system("mkdir -p %s" % out_dir)

    for i in range(0, len(merge_dirs)):
        merge_dir = merge_dirs[i]
        if detail:
            print("To merge %s" % merge_dir)
        file_names = os.listdir("%s/train/" % (merge_dir, ))
        for file_name in file_names:
            train_path = "%s/train/%s" % (merge_dir, file_name)
            test_path = "%s/test/%s" % (merge_dir, file_name)
            out_file = "%s/%s" % (out_dirs[i], file_name)
            os.system("cat %s >> %s; cat %s >> %s"
                      % (train_path, out_file, test_path, out_file))


def convert_propbank(detail=True):
    """
    Convert Wall Street Journal (wsj) to the input data combined with
    propbank
    """

    out_dir = "../data/show_key_words_wsj_propbank/"
    os.system("rm -rf %s" % (out_dir, ))
    os.system("mkdir -p %s" % (out_dir, ))
    show_key_words = True
    key_word_tag = "keywordtag"

    pb_instances = propbank.instances()
    # Count at first
    verb2idx = {}
    verb2frames = {}
    for i in range(0, len(pb_instances)):
        inst = pb_instances[i]
        verb_lemma, frame = inst.roleset.split(".")
        if verb_lemma not in verb2idx:
           verb2idx[verb_lemma] = []
        verb2idx[verb_lemma].append(i)
        if verb_lemma not in verb2frames:
            verb2frames[verb_lemma] = []
        if frame not in verb2frames[verb_lemma]:
            verb2frames[verb_lemma].append(frame)
    verb_nums = len(verb2idx.keys())
    verb_counter = 0

    pair_label = {'-LRB-':'(', '-RRB-':')', '-LCB-':'(', '-RCB-':')'}
    for verb_lemma, idxs in verb2idx.items():
        verb_counter += 1
        if len(verb2frames[verb_lemma]) < 2:
            continue
        fh = open("%s/%s" % (out_dir, verb_lemma), "w")
        if detail:
            print("processing %s(%s/%s)"
                  % (verb_lemma, verb_counter, verb_nums))
        for i in idxs:
            inst = pb_instances[i]
            fileid = inst.fileid
            sent_num = inst.sentnum
            verb_pos = inst.wordnum
            arguments = inst.arguments
            argument_word_pos = []
            for argloc, argid in arguments:
                # pos is something like "22:1,24:0,25:1*27:0"
                pos = str(argloc)
                # After relacing pos is something like "22:1 24:0 25:1 27:0"
                pos = pos.replace(",", " ").replace("*", " ")
                # Single item, e.g., 22:1
                if pos.find(" ") < 0:
                    items = [pos]
                else:
                    items = pos.split(" ")
                # Multiple items
                for item in items:
                    wordnum, height = item.split(":")
                    wordnum = int(wordnum)
                    height = int(height)
                    pos_list = [
                        x for x in range(wordnum, wordnum + height + 1)
                    ]
                    argument_word_pos.extend(pos_list)
            # Add verb pos. Verb itself is the argument
            argument_word_pos.append(verb_pos)
            # Remove duplicate pos
            argument_word_pos = list(set(argument_word_pos))

            verb_lemma, frame = inst.roleset.split(".")
            section = [x for x in fileid if x.isdigit()][0:2]
            section = "".join(section)
            fileid_for_ptb = "WSJ/%s/%s" % (section, fileid.upper())

            tagged_sent = ptb.tagged_sents(fileid_for_ptb)[sent_num]
            # Change tagged_sent from [tuples] to [list]
            tagged_sent = [[x[0], x[1]]for x in tagged_sent]

            # Show key words
            if show_key_words:
                for word_pos in range(0, len(tagged_sent)):
                    if word_pos not in argument_word_pos:
                        continue
                    word = tagged_sent[word_pos][0]
                    if (word.find("*") >=0
                        or word.lower() in ["the", "a", "an"]
                        or word in string.punctuation or word == "``"):
                        continue
                    tagged_sent[word_pos][0] += key_word_tag

            verb_bak = tagged_sent[verb_pos][0]
            verb_identifier = "verb_identifier_xxxxx"
            tagged_sent[verb_pos][0] = verb_identifier
            sent = []
            for (token, tag)in tagged_sent:
                if tag != '-NONE-':
                    if token in pair_label:
                        token = pair_label[token]
                    sent.append(token)
            sent = " ".join(sent)
            sent_toks = nltk.sent_tokenize(sent)
            candidate_sent = None
            for sent_tok in sent_toks:
                if sent_tok.find(verb_identifier) >= 0:
                    candidate_sent = sent_tok
            left_sent, right_sent = candidate_sent.split(verb_identifier)
            left_sent = left_sent.strip()
            right_sent = right_sent.strip()
            out_line = "%s\t%s\t%s\t%s" % (frame, left_sent, verb_bak, right_sent)
            out_line = remove_punctuations(out_line)
            print(out_line, file=fh)
        fh.close()


def load_semlink(detail=True):
    """
    Load Semlink WSJ files
    """

    # Load wsjTokens in semlink (sl)
    sl_wsj_path = "../../../nltk_data/corpora/semlink1.2.2c/wsjTokens/"
    # wsj sections
    wsj_secs = os.listdir(sl_wsj_path)
    sl_wsj_labels = {}
    sentid_labelidx_map = {}
    for wsj_sec in wsj_secs:
        wsj_sec_path = "%s/%s" % (sl_wsj_path, wsj_sec)
        sec_files = os.listdir(wsj_sec_path)
        if detail:
            print("wsj section: %s" % (wsj_sec, ))
        for sec_file in sec_files:
            sec_file_path = "%s/%s" % (wsj_sec_path, sec_file)
            # Skip hide file, e.g., ".xx.swp"
            if sec_file.startswith("."):
                continue
            if detail:
                print("file: %s" % sec_file)

            doc = ("WSJ/%s/%s.MRG"
                   % (wsj_sec, sec_file.replace(".sl", "").upper()))
            if doc not in sl_wsj_labels:
                sl_wsj_labels[doc] = []
            if doc not in sentid_labelidx_map:
                sentid_labelidx_map[doc] = {}

            fh = open(sec_file_path, "r")
            for line in fh:
                line = line.strip()
                # Skip empty lines
                if line == '':
                    continue
                items = line.split()
                sent_id = items[1]
                verb_pos = items[2]
                verb = items[4].split("-")[0]
                verbnet_class = items[5]
                framenet_frame = items[6]
                pb_grouping = items[7]
                si_grouping = items[8]

                # Find arguments of target verb
                arguments = items[10:]
                # arguments, e.g., 0:1-ARG0=Agent 3:0-rel 6:2-ARG1=Topic
                argument_list = []
                for argument in arguments:
                    pos_labels = argument.split("-");
                    pos = pos_labels[0]
                    label = pos_labels[1]
                    if label == "rel":
                        continue
                    pos = pos.replace(",", " ").replace("*", " ")
                    pos = pos.replace(";", " ")
                    # Single item, e.g., 22:1
                    if pos.find(" ") < 0:
                        items = [pos]
                    else:
                        items = pos.split(" ")
                    # Multiple items
                    for item in items:
                        wordnum, height = item.split(":")
                        wordnum = int(wordnum)
                        height = int(height)
                        pos_list = [
                            x for x in range(wordnum, wordnum + height + 1)
                        ]
                        argument_list.extend(pos_list)
                # The following line is commented because this verb_pos
                # sometimes is different to verb_pos in propbank (now I used)
                #  argument_list.append(int(verb_pos))
                # Remove duplicate
                argument_list = list(set(argument_list))

                sl_wsj_labels[doc].append([sent_id, verb_pos, verb, verbnet_class,
                                          framenet_frame, pb_grouping,
                                          si_grouping, argument_list])
                sentid_labelidx_map[doc]["%s_%s" % (sent_id, verb)] = (
                    len(sl_wsj_labels[doc]) - 1
                )
            fh.close()

    return sl_wsj_labels, sentid_labelidx_map


def convert_semlink_wsj2(detail=True):
    """
    (Current version of Semlink have some problems to be fixed.)
    This version 2 make use of propbank annotation on penn treebank. It will
    use the word number(in PropBank) to index in Semlink WSJ files instead of
    using token number(in Semlink WSJ files)
    """

    sl_wsj_labels, sentid_labelidx_map = load_semlink()
    sl_counters = summary_semlink_wsj(sl_wsj_labels, is_print=False)

    show_key_words = True
    key_word_tag = "keywordtag"

    out_dirs = ["../data/show_key_words_wsj_framnet/",
                "../data/show_key_words_wsj_verbnet/",
                "../data/show_key_words_wsj_sense"]
    sents_thresholds = [300, 300, 300]
    out_files = ["wsj.framenet", "wsj.verbnet", "wsj.sense"]
    frame_indexs = [4, 3, 6]
    corpus_names = ["framenet_frame", "verbnet_class", "si_grouping"]
    excludes = [2]
    for t in range(0, len(out_dirs)):
        if t in excludes:
            continue
        out_dir = out_dirs[t]
        os.system("rm -rf %s" % (out_dir, ))
        os.system("mkdir -p %s" % (out_dir, ))


    pb_instances = propbank.instances()
    pair_label = {'-LRB-':'(', '-RRB-':')', '-LCB-':'(', '-RCB-':')'}

    for t in range(0, len(out_dirs)):
        if t in excludes:
            continue
        if detail:
            print("To process %s" % (corpus_names[t]))

        out_dir = out_dirs[t]
        out_file = out_files[t]
        sents_threshold = sents_thresholds[t]
        fh = open("%s/%s" % (out_dir, out_file), "w")

        for i in range(0, len(pb_instances)):
            inst = pb_instances[i]
            fileid = inst.fileid
            sent_num = inst.sentnum
            verb_pos = inst.wordnum
            verb_lemma, _ = inst.roleset.split(".")
            section = [x for x in fileid if x.isdigit()][0:2]
            section = "".join(section)
            fileid_for_ptb = "WSJ/%s/%s" % (section, fileid.upper())
            key = "%s_%s" % (sent_num, verb_lemma)
            # Annotation in propbank not exists in Semlink
            if fileid_for_ptb not in sl_wsj_labels:
                continue
            # Labelled instance in PropBank not exist in Semlink WSJ files
            if key not in sentid_labelidx_map[fileid_for_ptb]:
                continue
            sl_idx = sentid_labelidx_map[fileid_for_ptb][key]
            sl_taginfo = sl_wsj_labels[fileid_for_ptb][sl_idx]
            frame_index = frame_indexs[t]
            frame = sl_taginfo[frame_index]
            # Too little sentences
            corpus_name = corpus_names[t]
            if (sl_counters[corpus_name][frame][0]
                <= sents_thresholds[t]):
                continue

            tagged_sent = ptb.tagged_sents(fileid_for_ptb)[sent_num]
            # Change tagged_sent from [tuples] to [list]
            tagged_sent = [[x[0], x[1]]for x in tagged_sent]

            # Show key words
            if show_key_words:
                argument_list = sl_taginfo[7]
                # Add verb pos if it doesn't have
                if verb_pos not in argument_list:
                    argument_list.append(verb_pos)
                for word_pos in range(0, len(tagged_sent)):
                    if word_pos not in argument_list:
                        continue
                    word = tagged_sent[word_pos][0]
                    if (word.find("*") >=0
                        or word.lower() in ["the", "a", "an"]
                        or word in string.punctuation or word == "``"):
                        continue
                    tagged_sent[word_pos][0] += key_word_tag

            verb_bak = tagged_sent[verb_pos][0]
            verb_identifier = "verb_identifier_xxxxx"
            tagged_sent[verb_pos][0] = verb_identifier
            sent = []
            for (token, tag)in tagged_sent:
                if tag != '-NONE-':
                    if token in pair_label:
                        token = pair_label[token]
                    sent.append(token)
            sent = " ".join(sent)
            sent_toks = nltk.sent_tokenize(sent)
            candidate_sent = None
            for sent_tok in sent_toks:
                if sent_tok.find(verb_identifier) >= 0:
                    candidate_sent = sent_tok
            left_sent, right_sent = candidate_sent.split(verb_identifier)
            left_sent = left_sent.strip()
            right_sent = right_sent.strip()
            out_line = ("%s\t%s\t%s\t%s"
                        % (frame, left_sent, verb_bak, right_sent))
            out_line = remove_punctuations(out_line)
            print(out_line, file=fh)
        fh.close()

def convert_semlink_wsj(detail=True):
    """
    (Current version of Semlink have some problems to be fixed. This function
    does not correctly convert the wsj to the input data)
    Convert Wall Street Journal (wsj) to the input data combined with SemLink.
    It will generate three dataset:
        1. wsj corpus labelled by PropBank
        2. wsj corpus labelled by FrameNet
        3. wsj corpus labelled by VerbNet
    """

    sl_wsj_labels = load_semlink()

    # print_semlink_wsj(sl_wsj_labels)
    wnl = WordNetLemmatizer()

    # Generate labelled data (framenet label)
    recover_label = {'-LRB-':'(', '-RRB-':')', '-LCB-':'(', '-RCB-':')'}
    framnet_out_dir = "../data/wsj_framnet/"
    framnet_out_file = "%s/wsj_label.framnet" % (framnet_out_dir, )
    os.system("rm -rf %s" % (framnet_out_dir, ))
    os.system("mkdir -p %s" % (framnet_out_dir, ))
    fh = open(framnet_out_file, "w")
    for doc_name in sl_wsj_labels.keys():
        sents = ptb.sents(doc_name)
        # sents = ptb.tagged_sents(doc_name)
        doc = sl_wsj_labels[doc_name]
        for i in range(0, len(doc)):
            sent_idx = int(doc[i][0])
            frame = doc[i][4]
            # Not used because some error correspondings between semlink and
            # penn treebank corpus.
            verb_pos = int(doc[i][1])
            # verb_pos = -1
            verb_lemma = doc[i][2]
            sent = sents[sent_idx]
            # sent = []
            # for tokens in sents[sent_idx]:
                # if tokens[1] != '-NONE-':
                    # token = tokens[0]
                    # if tokens[0] in recover_label:
                        # token = recover_label[token]
                    # if wnl.lemmatize(token, pos='v') == verb_lemma:
                        # verb_pos = len(sent)
                    # sent.append(token)
            # if verb_pos == -1:
                # print("\n")
                # print("doc_name:%s\tdoc[%s]:%s" % (doc_name, i, doc[i]))
                # print("len(sent):%s, sent:%s" % (len(sent), sent))
            if verb_pos >= len(sent):
                print("\n")
                print("doc_name:%s\tdoc[%s]:%s" % (doc_name, i, doc[i]))
                print("len(sent):%s, sent:%s" % (len(sent), sent))
                continue

            verb = sent[verb_pos]

            left_sent = " ".join(sent[0:verb_pos])
            right_sent = " ".join(sent[verb_pos + 1:])
            out_line = "%s\t%s\t%s\t%s" % (frame, left_sent, verb, right_sent)
            out_line = remove_punctuations(out_line)
            print(out_line, file=fh)
    fh.close()


def summary_semlink_wsj(sl_wsj_labels, is_print=True):
    """
    The summary statistics of semlink labels on wsj corpus
    """

    counters = {
        "verbnet_class":{}, "framenet_frame":{}, "pb_grouping":{},
        "si_grouping":{}
    }
    for doc_name in sl_wsj_labels.keys():
        doc = sl_wsj_labels[doc_name]
        for i in range(0, len(doc)):
            field_idxs = [3, 4, 5, 6]
            field_names = ["verbnet_class", "framenet_frame", "pb_grouping",
                           "si_grouping"]
            for field_idx, field_name in zip(field_idxs, field_names):
                if doc[i][field_idx] not in counters[field_name]:
                    counters[field_name][doc[i][field_idx]] = [0, {}]
                counters[field_name][doc[i][field_idx]][0] += 1
                if doc[i][2] not in counters[field_name][doc[i][field_idx]][1]:
                    counters[field_name][doc[i][field_idx]][1][doc[i][2]] = 0
                counters[field_name][doc[i][field_idx]][1][doc[i][2]] += 1

    if is_print:
        for field_name, counter_info in counters.items():
            print("######%s:\t\t%s" % (field_name, len(counter_info.keys())))
            for label, number in counter_info.items():
                print("%s:\t%s verbs, %s instances, \t%s"
                      % (label, len(number[1].keys()), number[0], number[1]))

    return counters

def convert_chn_text(detail=True):
    """
    Convert Chinese annotated text to the required format. The Chinese text
    should be utf-8 encoding
    """
    p = {
        "data_path": "../data/data_literature",
        "output_dir": "../data/converted_data"
    }
    if detail:
        gen_params_info(p)

    os.system("rm -rf %s" % p["output_dir"])
    os.system("mkdir -p %s" % p["output_dir"])
    files = os.listdir(p["data_path"])
    for file_name in files:
        if detail:
            print("to process %s" % file_name)
        file_path = "%s/%s" % (p["data_path"], file_name)
        out_file_path = "%s/%s" % (p["output_dir"], file_name)
        fh_in = codecs.open(filename=file_path, mode="r", encoding='utf8')
        fh_out = codecs.open(filename=out_file_path, mode="w", encoding='utf8')
        line_idx = 1
        verb = ""
        for line in fh_in:
            line = line.lstrip()
            if line.find("\t") < 0:
                print("Please check in file %s, line: %s\nsentence :%s\n"\
                    "The above sentence has NO TAB and has been skiped!" \
                        % (file_name, line_idx, line))
                continue
            items = line.split("\t")
            if len(items) != 4:
                print("Please check in file %s, line: %s\nsentence :%s\n"\
                    "The above sentence has NO 4 TAB and has been skiped!" \
                        % (file_name, line_idx, line))
                continue
            frame_id = items[0]
            if frame_id.find(".") >= 0:
                frame_id = frame_id.split(".")[0]
            verb = items[2].strip()
            left_sent = items[1].strip()
            right_sent = items[3].strip()
            out_line = "%s\t%s\t%s\t%s"\
                    % (frame_id, left_sent, verb, right_sent)
            print(out_line, file=fh_out)

            line_idx += 1

        fh_in.close()
        fh_out.close()

def convert_semeval_without_extraction(detail=True):
    """
    Convert semeval data without extraction the arugments of target verb
    """

    # Parameters
    p = {
        "data_sets": ["../cpa_data/Microcheck/", "../cpa_data/testdata/Microcheck"],
        "output_dir": "../data/semeval_mic_train_and_test_no_extraction"
    }
    #  if detail:
        #  print_params(p)
    os.system("rm -rf %s" % p["output_dir"])
    os.system("mkdir -p %s" % p["output_dir"])
    for data_set_path in p["data_sets"]:
        files = os.listdir("%s/input/" % data_set_path)
        for file_name in files:
            if detail:
                print("to process %s" % file_name)
            verb_name, ext = os.path.splitext(file_name)
            file_path = "%s/input/%s" % (data_set_path, file_name)
            # Read cluster
            file_cluster_path = "%s/task2/%s.clust" % (data_set_path, verb_name)
            fh = open(file_cluster_path, "r")
            verb_id2cluster_id = {}
            for line in fh:
                line = line.strip()
                if line == "":
                    continue
                verb_id, cluster_id = line.split("\t")
                verb_id2cluster_id[verb_id] = cluster_id
            fh.close()

            out_file_path = "%s/%s" % (p["output_dir"], verb_name)
            fh_out = open(out_file_path, "w")
            fh = open(file_path, "r", encoding = "ISO-8859-1")
            sent = ""
            cluster_id = -1
            for line in fh:
                line = line.strip()
                if line == "":
                    fh_out.write("%s\t%s\n" % (cluster_id, remove_punctuations(sent)))
                    sent = ""
                    continue
                tokens = line.split("\t")
                if len(tokens) == 3 and tokens[2] == "v":
                    sent += "\t%s\t" % tokens[1]
                    cluster_id = verb_id2cluster_id[tokens[0]]
                else:
                    sent += tokens[1] + " "
            fh.close()
            fh_out.close()

def convert_semeval_with_extraction(detail=True):
    """
    Convert semeval data with extraction the arugments of target verb
    """

    # Parameters
    p = {
        "data_sets": ["../data/semeval2015_task15/train/Microcheck/", "../data/semeval2015_task15/test/Microcheck/"],
        "output_dir": "../data/semeval_mic_train_and_test_with_extraction",
        "relations": ["subj", "obj", "iobj", "advprep", "acomp", "scomp"],
    }
    #  if detail:
        #  print_params(p)
    os.system("rm -rf %s" % p["output_dir"])
    os.system("mkdir -p %s" % p["output_dir"])
    for data_set_path in p["data_sets"]:
        files = os.listdir("%s/task1/" % data_set_path)
        for file_name in files:
            if detail:
                print("to process %s" % file_name)
            verb_name, ext = os.path.splitext(file_name)
            file_path = "%s/task1/%s" % (data_set_path, file_name)
            # Read cluster
            file_cluster_path = "%s/task2/%s.clust" % (data_set_path, verb_name)
            fh = open(file_cluster_path, "r")
            verb_id2cluster_id = {}
            for line in fh:
                line = line.strip()
                if line == "":
                    continue
                verb_id, cluster_id = line.split("\t")
                verb_id2cluster_id[verb_id] = cluster_id
            fh.close()

            out_file_path = "%s/%s" % (p["output_dir"], verb_name)
            fh_out = open(out_file_path, "w")
            fh = open(file_path, "r", encoding = "ISO-8859-1")
            sent = ""
            cluster_id = -1
            for line in fh:
                line = line.strip()
                if line == "":
                    fh_out.write("%s\t%s\n" % (cluster_id, sent))
                    sent = ""
                    continue
                tokens = line.split("\t")
                if len(tokens) == 4:
                    if tokens[2] == "v":
                        sent += "\t%s\t" % tokens[1]
                        cluster_id = verb_id2cluster_id[tokens[0]]
                    else:
                        sent += tokens[1] + " "

            fh.close()
            fh_out.close()

def convert_semeval_with_key_words_showing(detail=True):
    """
    Convert semeval data with key words showing
    """

    # Parameters
    p = {
        "data_sets": ["../data/semeval2015_task15/train/Microcheck/", "../data/semeval2015_task15/test/Microcheck/"],
        "output_dir": "../data/semeval_mic_train_and_test_with_key_words",
        "relations": ["subj", "obj", "iobj", "advprep", "acomp", "scomp"],
        "key_words_tag": "keywordtag"
    }
    #  if detail:
        #  print_params(p)
    os.system("rm -rf %s" % p["output_dir"])
    os.system("mkdir -p %s" % p["output_dir"])
    for data_set_path in p["data_sets"]:
        files = os.listdir("%s/task1/" % data_set_path)
        for file_name in files:
            if detail:
                print("to process %s" % file_name)
            verb_name, ext = os.path.splitext(file_name)
            file_path = "%s/task1/%s" % (data_set_path, file_name)
            # Read cluster
            file_cluster_path = "%s/task2/%s.clust" % (data_set_path, verb_name)
            fh = open(file_cluster_path, "r")
            verb_id2cluster_id = {}
            for line in fh:
                line = line.strip()
                if line == "":
                    continue
                verb_id, cluster_id = line.split("\t")
                verb_id2cluster_id[verb_id] = cluster_id
            fh.close()

            out_file_path = "%s/%s" % (p["output_dir"], verb_name)
            fh_out = open(out_file_path, "w")
            fh = open(file_path, "r", encoding = "ISO-8859-1")
            sent = ""
            cluster_id = -1
            for line in fh:
                line = line.strip()
                if line == "":
                    fh_out.write("%s\t%s\n" % (cluster_id, sent))
                    sent = ""
                    continue
                tokens = line.split("\t")
                if len(tokens) == 4:
                    if tokens[2] == "v":
                        sent += "\t%s%s\t" % (tokens[1], p["key_words_tag"])
                        cluster_id = verb_id2cluster_id[tokens[0]]
                    else:
                        sent += tokens[1] + p["key_words_tag"] + " "
                else:
                    sent += tokens[1] + " "

            fh.close()
            fh_out.close()

def convert_pdev(detail=True):
    """
    Convert pdev data
    """
    p = {
        "pdev_dir": "../split_pdev/train_pdev/",
        "output_dir": "../data/split_pdev/train"
    }
    os.system("rm %s -rf" % p["output_dir"])
    os.system("mkdir -p %s" % p["output_dir"])
    files = os.listdir(p["pdev_dir"])
    for file_name in files:
        file_path = "%s/%s" % (p["pdev_dir"], file_name)
        fh_in = open(file_path, "r")
        out_file = "%s/%s" % (p["output_dir"], file_name)
        fh_out = open(out_file, "w")
        for line in fh_in:
            line = line.strip()
            if line == "":
                continue
            line = html.unescape(line).replace("<p>", " . ").replace("</p>", " . ")
            items = line.split("\t")
            if len(items) != 4:
                continue
            frame = items[2].strip()
            # Processing frame
            frame = frame.split(".")[0]
            try:
                frame = int(frame)
            except:
                logging.warn("Skip frame: %s" % frame)
                continue
            verb = items[1].strip()
            left_sent = remove_punctuations(nltk.sent_tokenize(items[0])[-1])
            right_sent = remove_punctuations(nltk.sent_tokenize(items[3])[0])
            # Remove corpus title infomation
            right_sent = right_sent.replace("British National Corpus", "")
            fh_out.write("%d\t%s\t%s\t%s\n" % (frame, left_sent, verb, right_sent))
        fh_in.close()
        fh_out.close()

if __name__ == "__main__":
    # convert_semeval_without_extraction()
    #  convert_semeval_with_extraction()
    #  convert_semeval_with_key_words_showing()
    # convert_pdev()
    # convert_chn_text()
    #  convert_propbank()
    convert_semlink_wsj2()
    #  merge_split_data()
    #  convert_chn_propbank()

