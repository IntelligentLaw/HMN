#coding=utf8
import json
from utils import data_helper
import math

def parse_line_from_file(file_path, label1_name = "parent_class", label2_name="laws"):
    with open(file_path, 'r') as f:
        line = f.read().rstrip()
        text, textlens, label1, label2 = parse_line(line, label1_name, label2_name)
        return text, textlens, label1, label2

        # accu = json_line['accu']
        # laws = json_line['laws']
        # imp = json_line['imp']

def parse_line(line, label1_name, label2_name):
    json_line = json.loads(line)
    text = json_line['textIds']
    textlens = json_line['text_len']
    label2 = json_line[label2_name]
    label1 = json_line[label1_name]

    # train_label = json_line['train_label']
    return text, textlens, label1, label2


if __name__ == "__main__":
    data_path = "../new_data/preprocess_good/transform_data_train.json"
    law_path = "../new_data/law_dict.pkl"
    accu_path = "../new_data/accu_dict.pkl"
    parent_path = "new_data_with_parent/parent_dict.pkl"

    law_Helper = data_helper.OneHotEncoding(law_path)
    accu_Helper = data_helper.OneHotEncoding(accu_path)
    parent_Helper = data_helper.OneHotEncoding(parent_path)
    with open(data_path, 'r') as f:
        data = f.readlines()
        for line in data:
            text, label, textlens, parent, parentlens = parse_line(line, 'parent_class')
            print(text)
            print(label)
            print(textlens)
            print(law_Helper.transform_raw(label))

            break




