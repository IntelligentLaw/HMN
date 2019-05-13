# coding=utf-8
import torch
import torch.nn as nn
import numpy as np
import sys
import json
def get_data(data_list):
    source_list = list()
    target_list = list()
    for [key, value] in data_list:
        source_list.append(key)
        target_list.append(value)
    print('\t'.join(map(str, source_list)))
    print('\t'.join(map(str, target_list)))

if __name__ == "__main__":
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    data_size = "small"
    data_type = "valid"
    with open("../statistic_result/" + data_size + "_"+ data_type +"_result.txt", 'r') as f:
        data = f.read().rstrip()
        json_line = json.loads(data)

        print(json_line.keys())
        laws_list = json_line['laws']
        accu_list = json_line['accu']
        imp_list = json_line['imp']
        avg_len = json_line['len']
        max_len = json_line['max_len']
        min_len = json_line['min_len']


        print("laws")
        get_data(laws_list)
        print("accu")
        get_data(accu_list)
        print("imp")
        get_data(imp_list)
        print("avg len")
        print(avg_len)
        print("max len")
        print(max_len)
        print("min len")
        print(min_len)