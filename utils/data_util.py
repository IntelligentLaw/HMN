import pickle

import numpy as np
import random
import torch
from torch.utils import data

from utils.clean_arg import data_file


def glove(word2id, glove_f):
    with open(word2id,"r",encoding='utf-8') as w:
        word2 = {}
        lines = w.readlines()
        for line in lines:
            line = line.split("\t")
            word2[line[0]] = line[1]
        #print(len(lines))
        a = np.random.random((len(lines), 300))
        with open(glove_f,"r",encoding='utf-8') as g:
            word = g.readline()
            while word:
               # flag = False
                word = word.split(' ')
                if word[0] in word2.keys():
                    a[int(word2[word[0]])] = np.array(word[1:])
                    # print(np.array(word[1:]))
                word = g.readline()
        return a

def get_id2word(file):
    id2word = {}
    with open(file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            id2word[int(line[1])] = line[0]
    return id2word

def process_data(data_count_each_file, data_count_last_file , file_count):
    data_list = []
    for i in range(file_count-1):
        for j in range(data_count_each_file):
            data_list.append((str(i), str(j)))
    for j in range(data_count_last_file):
        data_list.append((str(file_count-1), j))
    return data_list

def batch_collate(batch):
    """
    对每一个batch的处理，将一个batch的数据对齐到该batch的最大长度
    :param batch: 一个batch的数据
    :return: 一个batch对齐后的文本的词语id向量数组, 类别标签数组
    """
    label, text = zip(*batch)
    word_ids, char_ids = zip(*text)
    #seq_lengths = torch.LongTensor([x.size for x in text])
    seq_lengths = torch.LongTensor([x.size for x in word_ids])
    #print(seq_lengths)
    #x = torch.zeros((len(batch), seq_lengths.max())).long()
    """
    change
    """
    x = torch.zeros((len(batch), 1500)).long()
    x_c = torch.zeros((len(batch), 1500, 16)).long()
    #for idx, (seq, seqlen) in enumerate(zip(text, seq_lengths)):
    for idx, (seq, seqlen) in enumerate(zip(word_ids, seq_lengths)):
        if seq.size != 0:
            x[idx, :seqlen] = torch.from_numpy(seq)
            x_c[idx, :seqlen, :] = torch.from_numpy(char_ids[idx])
            #print(char_ids[idx])
    #print(x_c)
    return x, x_c, torch.LongTensor(label)

def load_data(data_list, args):
    """
    生成训练集和测试集
    :param file2label: 文件名到标签的映射
    :param word2id:词语到id的映射
    :param args:模型参数，test_ratio，max_len，batch_size，shuffle
    :return: 训练集和测试集的DataLoader
    """
    print("\nLoading data...")
    # train_data, test_data = split_data(data_list, args.test_ratio, True)
    # with open('./data/data.pkl', 'wb') as f:
    #     pickle.dump(train_data, f)
    #     pickle.dump(test_data, f)

    with open('./data/data.pkl', 'rb') as f:
        train_data = pickle.load(f)
        test_data = pickle.load(f)

    train_dataset = TextData(train_data, max_len=args.max_len, class_num=args.class_num)
    test_dataset = TextData(test_data, max_len=args.max_len, class_num=args.class_num)
    train_iter = data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        collate_fn=batch_collate,
        num_workers=12
    )
    test_iter = data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=batch_collate,
        num_workers=12
    )
    return train_iter, test_iter

def split_data(data_list, test_ratio=0.1, shuffle=True):
    """
    划分训练集和测试集
    :param data_list:所有数据，本模型中为所有文件名到label的列表
    :param test_ratio:测试集比例
    :param shuffle:是否随机重新排列数据
    :return:返回训练集和测试集
    """
    if shuffle:
        random.shuffle(data_list)
    test_index = int((1.-test_ratio) * len(data_list))
    return data_list[:test_index], data_list[test_index:]
"""
change
"""
def text2idlist(data_pos, id2word, max_len=-1, class_num = 13000):
    """
    从分好词的文本文件生成词语id列表，并作截断处理，这里并不做补齐处理，因为之后处理一个batch的数据时对齐到bacth的最长
    :param file_name:分好词的文本文件的文件名
    :param max_len: 指定文本文件的最大长度，小于1时不做截断处理，
                    大于1时:
                    如果长度小于等于max_len，不处理；
                    如果长度大于max_len，截断到max_len
                    这样处理是为了防止有的文本特别长，可能到几十万个词，会占用比较大的内存，导致内存溢出
    :return:一个文本文件对应的词语id列表
    """
    file_name = data_file + data_pos[0] + '.txt'
    index = int(data_pos[1])
    with open(file_name, 'r', encoding='utf-8') as f:
        line = f.readlines()[index].split('\t')
        label_ids, word_ids = [int(item) for item in line[0].split(' ')], [int(item) for item in line[1].split(' ')]
        labels = [0] * class_num
        char_ids = np.zeros((max_len, 16), np.long)
        #print(line[1])
        for i, item in enumerate(line[1].split(' ')):
            if i >= max_len:
                break
            if int(item) <= 1:
                continue
            word = id2word[int(item)]
            for j, char in enumerate(word):
                if j >= 16:
                    break
                char_ids[i][j] = ord(char) - ord('a') + 1
        for item in label_ids:
            labels[item] = 1
        if max_len > 0 and len(word_ids) > max_len:
            word_ids = word_ids[:max_len]
        if len(word_ids) < max_len:
            word_ids.extend([0] * (max_len - len(word_ids)))
        return labels, word_ids, char_ids

class TextData(data.Dataset):
    """
    自定义的TextData类，继承data.Dataset
    需要重写 __init__, __getitem__, __len__3个函数
    """
    """
    change
    """
    def __init__(self, data_list, class_num, id2word=get_id2word('./vocab/word2id.txt'), transform=text2idlist,max_len=-1):
        """
        构造函数
        :param file2label: 文件名到标签的映射
        :param word2id: 词语到id的映射
        :param transform: 每条数据的转换操作函数
        :param max_len: 指定截断最大长度
        """
        self.data_list = data_list
        self.class_num = class_num
        """
        change
        """
        self.id2word = id2word
        self.transform = transform
        self.max_len = max_len
        self.len = len(data_list)

    def __getitem__(self, index):
        """
        根据index获取一条训练数据
        :param index: 数据的下标
        :return: 一条数据的特征及标签
        """
        data_pos = self.data_list[index]
        """
        change
        """
        label_ids, word_ids, char_ids = self.transform(data_pos, self.id2word, self.max_len, self.class_num)
        #print(np.array(idlist).size)
        return label_ids, (np.array(word_ids), np.array(char_ids))#torch.LongTensor(idlist), label

    def __len__(self):
        """
        获取数据条数
        :return: 数据条数
        """
        return self.len
