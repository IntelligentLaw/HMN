#coding=utf8
import os,sys
import json
import jieba
import torch
from utils.parse_data import *

stopWords =True
with open('/home/iot/fanyu/Attention-BiDAF/data/stop_words.txt', 'r') as stops:
    s = stops.read()
    stop_words_from_file = s.split()
stop_words_from_file = set(stop_words_from_file)
total_stop_words_set = stop_words_from_file

def stopword_remover(line):

    tokens = [w for w in line if not (w in total_stop_words_set
                    or is_number(w) or ((not is_ascii(w)) and len(w) <= 1))]
    return tokens

def is_ascii(s):
    return all(ord(c) < 128 for c in s)

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


def makelaw():
    """
    read the law file and use Word2vec to transform
    :return:law_text
    :return:law_length
    :return:law_order
    :return:parent2law(The type of law that each parent class contains)
    """

    filelaw = "/home/iot/fanyu/Attention-BiDAF/new_data_with_parent/newlaw.txt"
    input_lines = open(filelaw, 'r')
    filetext = "/home/iot/fanyu/Attention-BiDAF/new_data_with_parent/rulenew.txt"
    text = open(filetext, 'r')
    wordHelper = data_helper.Vocab("/home/iot/fanyu/Attention-BiDAF/new_data_with_parent/word_dict_10w.pkl")
    with open('/home/iot/fanyu/Attention-BiDAF/new_data_with_parent/my_dict.txt', 'r') as f:
        data = f.read()
        data = data.rstrip()
        my_dict = json.loads(data)

    parent2law = [[] for i in range(10)]
    law = []
    for line in input_lines:
        line = line.rstrip()
        law.append(line)
        parent2law[my_dict[str(line)]-1].append(int(line))

    law.sort()

    for part in parent2law:
        part.sort()

    i = 0
    law_text = []
    law_length = []
    law_order = []
    for line in text:
        line = line.rstrip()
        index, raw_text = line.split('\t')
        if index == law[i]:
            words = list(jieba.cut(raw_text))
            seg_fact = stopword_remover(words) if stopWords == True else words
            # print(index,"---",seg_fact[0:5])
            # print(index, "---", seg_fact[0:5])
            trans_text = wordHelper.transform_raw(seg_fact[0:5])
            length = len(trans_text)

            law_text.append(trans_text)
            law_length.append(length)
            law_order.append(int(index))
            if i + 1 == len(law):
                break
        else:
            continue
        i = i + 1
    maxlen = max(law_length)
    law_text = [item + [0] * (maxlen - len(item)) for item in law_text]

    return torch.LongTensor(law_text), torch.LongTensor(law_length), law_order, parent2law

if __name__ == "__main__":
    makelaw()