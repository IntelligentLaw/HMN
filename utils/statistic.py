import sys
import os
import json
def get_list_dict_result(my_dict, value_list):
    for value in value_list:
        my_dict[value] = my_dict.get(value, 0) + 1
    return my_dict
# with open("./new_data/preprocess_good/transform_data_train.json", 'r') as f:
#     data = f.readlines()

index = 0
sum_len = 0
max_len = 0
min_len = 2000
imp_dict = dict()
accu_dict = dict()
laws_dict = dict()

for line in sys.stdin:
    line = line.rstrip()
    if not line:
        break
    json_line = json.loads(line)

    imp = str(json_line['imp'])
    accu = json_line['accu']
    laws = json_line['laws']
    text_len = json_line['text_len']
    if text_len > max_len:
        max_len = text_len
    if text_len < min_len:
        min_len = text_len
    imp_dict[imp] = imp_dict.get(imp, 0) + 1
    accu_dict = get_list_dict_result(accu_dict, accu)
    laws_dict = get_list_dict_result(laws_dict, laws)
    sum_len += text_len
    index += 1
avg_len = sum_len/index

sort_imp = sorted(imp_dict.items(), key = lambda d:d[0])
sort_accu = sorted(accu_dict.items(), key = lambda d:d[0])
sort_laws = sorted(laws_dict.items(), key = lambda d:d[0])
my_dict = {'imp':sort_imp, 'accu':sort_accu, 'laws':sort_laws, 'len': avg_len, 'max_len': max_len, 'min_len':  min_len}

print(json.dumps(my_dict, ensure_ascii=False))


