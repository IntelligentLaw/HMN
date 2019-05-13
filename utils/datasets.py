#coding=utf8
import os
import json
import torch
import torchtext.data as data
from torch.autograd import Variable
from utils import parse_data
import torchtext.datasets as datasets
from utils import data_helper
from utils.parse_data import *
from utils.parse_traindata import *
import sys
import os
o_path = os.getcwd() # 返回当前工作目录
sys.path.append(o_path) # 添加自己指定的搜索路径

def make_data(train_data_path, dev_data_path, law_path, parent_path, word_dict_path, batch_size,dev_batch_size):

    words_Helper = data_helper.Vocab(word_dict_path)
    law_Helper = data_helper.OneHotEncoding(law_path)
    parent_Helper = data_helper.OneHotEncoding(parent_path)
    # p2law_Helper = data_helper
    word_num = len(words_Helper._word_to_id)+1
    law_num = len(law_Helper._word_to_id)
    parent_num = len(parent_Helper._word_to_id)
    train_dataset_helper = LawDataSet(train_data_path, loader=parse_line_from_file, transform=None,
                                      label1_transform=parent_Helper.transform_raw, label2_transform=law_Helper.transform_raw,flags = 0)
    dev_dataset_helper = LawDataSet(dev_data_path, loader=parse_line_from_file, transform=None,
                                    label1_transform=parent_Helper.transform_raw,label2_transform=law_Helper.transform_raw, flags = 0)
    train_iter = torch.utils.data.DataLoader(train_dataset_helper, batch_size=batch_size,
                                               shuffle=True, num_workers=8, collate_fn=my_collate)
    dev_iter = torch.utils.data.DataLoader(dev_dataset_helper, batch_size=dev_batch_size,
                                             shuffle=False, num_workers=8, collate_fn=my_collate)

    return train_iter, dev_iter, word_num, law_num, parent_num

def my_collate(batch):

    text = [item[0] for item in batch]
    label1 = [item[2] for item in batch]
    label2 = [item[3] for item in batch]
    law = [item[4] for item in batch]
    text_lens = [len(item[0]) for item in batch]
    max_len = max(text_lens)

    label2 = torch.FloatTensor(label2)
    laws = torch.nonzero(label2)[:,1]
    text = [item + [0] * (max_len - len(item)) for item in text]

    text = torch.LongTensor(text)[:,:10]
    for i,lenth in enumerate(text_lens):
        if lenth>10:
            text_lens[i]=10
    return [text, torch.LongTensor(text_lens), torch.FloatTensor(label1)[:,1:9],label2, laws]

    # print("#"*100)
    # print(text)
    # print(label)
    # print(text_lens)
# article_set = set()
# with open("./laws.txt", 'r') as f:
#     data = f.readlines()
#     for line in data:
#         line = line.rstrip()
#         article_set.add(line)
# article_list = list(article_set)

class LawDataSet(data.Dataset):
    def __init__(self, root, train=True, transform=None, label1_transform=None, label2_transform=None,
                 loader=None, pad_len=100, pad_parent_len=20, flags = 0):
        self.pad_len = pad_len
        self.pad_parent_len = pad_parent_len
        self.root = root
        self.train = train
        self.transform = transform
        self.label1_transform = label1_transform
        self.label2_transform = label2_transform
        self.loader = loader
        self.file_name_list = self.get_file_names()
        self.flag = flags

    def get_file_names(self):
        for root, dirs, files in os.walk(self.root):
            # print(files)
            return files

    def __getitem__(self, index):
        #处理一个样本
        data_path = self.root + self.file_name_list[index]
        text, textlens, label1, label2 = self.loader(data_path)


        if self.transform is not None:
            text = self.transform(text, length = self.pad_len)

        # [3,4,5,6] =  [1,2,3,4,5,6]  [1,2]
        # a = article_list - label1
        # index = np.radom.randint(len(a))
        # label_neg = [a[index]]

        if self.label1_transform is not None:
            label1 = self.label1_transform(label1)

        law = label2
        # label1_neg = self.label1_transform(label_neg)

        if self.label2_transform is not None:
            label2 = self.label2_transform(label2)

        return text, textlens, label1, label2, law

    def __len__(self):
        return len(self.file_name_list)



if __name__ == "__main__":


    accu_path = "../new_data/accu_dict.pkl"
    # law_path = "../new_data/law_dict.pkl"
    # data_path = "../new_data/preprocess_good/transform_data_train.json"
    # words_Helper = data_helper.Vocab()
    # laws_Helper = data_helper.OneHotEncoding(law_path)
    # accu_Helper = data_helper.OneHotEncoding(accu_path)
    #
    # dataset_helper = LawDataSet("../new_data/preprocess_good/transform_data_train_dir/",
    #                             loader=parse_line_from_file, transform=None,
    #                             label_transform=accu_Helper.transform_raw)
    # train_loader = torch.model_utils.data.DataLoader(dataset_helper, batch_size=3,
    #                                            shuffle=True, num_workers=8, collate_fn=my_collate)
    #
    # # text, label, lens = train_loader.__iter__().__next__()
    #
    # for batch_id, batch in enumerate(train_loader):
    #     print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    #     (text, label), lens = batch
    #     print(text)
    #     print(label)
    #     print(lens)
    #     break

