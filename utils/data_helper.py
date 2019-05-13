#coding=utf8
from collections import Counter
import itertools
import pickle
import os
import sys
o_path = os.getcwd() # 返回当前工作目录
sys.path.append(o_path) # 添加自己指定的搜索路径


class Vocab(object):
    def __init__(self, dict_path = None):
        self._word_to_id = {}
        self._id_to_word = {}

        self._word_to_id['<pad>'] = 0
        self._word_to_id['<unk>'] = 1
        self._id_to_word[0] = '<pad>'
        self._id_to_word[1] = '<unk>'
        self._count = 2
        if dict_path != None:
            self.load_dict(dict_path)

    def create_vocabulary_from_file(self, data_path, min_frequence = 1, max_size = 100000):
        with open(data_path, 'r') as vocab_f:
            for line in vocab_f:
                pieces = line.split('\t')
                if len(pieces) != 2:
                    print("Warning: incorrectly formatted line in vocabulary file: %s\n"%line)
                    continue
                w = pieces[0]
                count = pieces[1]
                if w in self._word_to_id:
                    raise Exception("Duplicated word in vocabulary file: %s"%w)
                if int(count) < min_frequence:
                    continue
                if len(self._word_to_id) == max_size:
                    break

                self._word_to_id[w] = self._count
                self._id_to_word[self._count] = w
                self._count = 1
            print("Finished contructing vocabulary of %i tootal words. Last word added: %s"%(len(self._word_to_id), self._id_to_word[self._count - 1]))

    def add_counter(self, data_list, counter = None):
        if counter == None:
            count_data = Counter(itertools.chain(*data_list))
            return count_data

    def update_counter(self, data_list, counter):
        data_iter = itertools.chain(*data_list)
        counter.update(data_iter)

    def create_vocabulary_from_counter(self, counter, min_frequence = 2, max_size = 100000):
        data_dict = dict(counter.most_common(max_size))
        j = 1
        sorted_dict = sorted(data_dict.items(), key=lambda x: x[1], reverse=True)
        for w, count in sorted_dict:
            if w in self._word_to_id:
                raise Exception("Duplicated word in vocabulary file: %S")
            if int(count) < min_frequence:
                continue

            if self._count >= max_size:
                break
            self._word_to_id[w] = self._count
            self._id_to_word[self._count] = w
            self._count += 1
        print("Finished constructing vocabulary of %i total words. Last word added: %s, the data count is %d" % (
            len(self._word_to_id), self._id_to_word[self._count - 1], count))

    def create_vocabulary_from_data(self, data, max_size = 200000):
        counter = Counter(itertools.chain(*data))
        data_dict = dict(counter.most_common(max_size))
        for w, count in data_dict.items():
            if w in self._word_to_id:
                raise Exception("Duplicated word in vocabulary file: %s"%w)

            if len(self._word_to_id) == max_size:
                break

            self._word_to_id[w] = self._count
            self._id_to_word[self._count] = w
            self._count += 1

        print("Finished constructing vocabulary of %i total words. Last word added: %s" % (
            len(self._word_to_id), self._id_to_word[self._count - 1]))

    def word2id(self, word):
        if word in self._word_to_id:
            return self._word_to_id[word]
        else:
            # modify by yz, avoid the None value problem
            return None

    def id2word(self, word_id):
        if word_id not in self._id_to_word:
            raise("id is not in vocab")
        else:
            return self._id_to_word[word_id]

    def Pad(self, ids, pad_id = 0, length = 500):
        assert pad_id is not None
        assert length is not None
        if len(ids) < length:
            pad_data = [pad_id] * (length - len(ids))
            return ids + pad_data
        else:
            return ids[:length]

    def transform_raw(self, text, pad_len = None, pad_id = 0):
        ids = []
        for w in text:
            if self.word2id(w) != None:
                ids.append(self.word2id(w))
        if pad_len is not None:
            return self.Pad(ids, pad_id, pad_len)
        return ids

    def save_dict(self, save_path):
        save_dict = {'word2id': self._word_to_id,
                     'id2word': self._id_to_word}
        with open(save_path, 'wb') as f:
            pickle.dump(save_dict, f)

    def load_dict(self, dict_path):
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0
        with open(dict_path, 'rb') as f:
            load_dict = pickle.load(f)
            self._word_to_id = load_dict['word2id']
            self._id_to_word = load_dict['id2word']
        self._count = len(self._word_to_id) - 1

class OneHotEncoding(object):
    def __init__(self, dict_path = None):
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0
        self.n_classes = 0
        if dict_path != None:
            self.load_dict(dict_path)

    def make_dict_from_file(self, data_path):
        with open(data_path, 'r') as f:
            for line in f:
                line = line.rstrip()
                if not line:
                    continue
                self._word_to_id[line] = self._count
                self._id_to_word[self._count] = line
                self._count += 1
            print("Finished constructing vocabulary of %i total words. Last word added: %s"%(self._count, self._id_to_word[self._count -1]))

    def make_dict_from_dict(self, data_dict):
        for w, count in data_dict.items():
            if w in self._word_to_id:
                raise Exception("Duplicated word in vocabulary file: %s"%w)

            self._word_to_id[w] = self._count
            self._id_to_word[self._count] = w
            self._count += 1

        print("Finished constructing vocabulary of %i total words. Last word added: %s"%(len(self._word_to_id), self._id_to_word[self._count -1]))
        self.n_classes = len(self._word_to_id)

    def add_value2dict(self, item):
        if item in self._word_to_id:
            raise Exception("Duplicated word in vocabulary file: %s" % item)
        self._word_to_id[item] = self._count
        self._id_to_word[self._count] = item
        self._count += 1

    def word2id(self, word):
        if word in self._word_to_id:
            return self._word_to_id[word]
        else:
            return None

    def id2word(self, word_id):
        if word_id not in self._id_to_word:
            raise('id is not in vocab')
    def transform_raw(self, label):
        res = [0] * self.n_classes
        for word in label:
            word = str(word)
            if self.word2id(word) != None:
                res[self.word2id(word)] = 1
        return res

    def save_dict(self, save_path):
        save_dict = {'word2id': self._word_to_id,
                     'id2word': self._id_to_word}
        with open(save_path, 'wb') as f:
            pickle.dump(save_dict, f)

    def load_dict(self, dict_path):
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0
        with open(dict_path, 'rb') as f:
            load_dict = pickle.load(f)
            self._word_to_id = load_dict['word2id']
            self._id_to_word = load_dict['id2word']
        self._count = len(self._word_to_id) - 1
        self.n_classes = len(self._word_to_id)

if __name__ == "__main__":
    print("~~~~~~~~~~")
    a = [['a', 'b', 'c'],
         ['c', 'e', 'd']]

    data = Counter(itertools.chain(*a))
    my_dict = dict(data)
    print(my_dict)
    print(max(my_dict.values()))
    print(list(itertools.chain(*a)))
