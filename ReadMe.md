## HMN
The implementation of HMN in our paper:Hierarchical Matching Network for Crime Classification



## Require
* Python 3.6


## Reproducing Results
* Run `python3 run.py`


## Dataset
CAIL2018, the download link:https://cail.oss-cn-qingdao.aliyuncs.com/CAIL2018_ALL_DATA.zip


## Example input data
the word has conveted into vector

* parent class:parent class
* laws:sub class
* textIds: the fact descriptions have been transformed from text to id

{"text_len": 41, "laws": [234],
"textIds": [2935, 10, 3, 330, 16, 406, 2935, 1802, 2, 272, 4328, 1064, 877, 818, 272, 5455,
9056, 41, 486, 192, 83, 430, 620, 27, 12, 31, 49, 2, 6, 346, 79, 9, 75, 7, 2, 69, 15, 66, 31,
30, 38], "parent_class": ["侵犯公民人身"]}


## Citing this repository
If you find this code useful in your research, please consider citing us:
```javascript
  @inproceedings{HMN_SIGIR_2019
    auther =  {PengFei Wang and Yu Fan}
    title  =  {Hierarchical Matching Network for Crime Classification}
    booktitle = {SIGIR}
    year =  {2019}
  }
```
