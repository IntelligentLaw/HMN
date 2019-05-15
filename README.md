## HMM
Source code for SIGIR 2019 paper "Hierarchical Matching Network for Crime Classification"

### Dependecies
* tqdm==4.31.1
* numpy==1.16.3
* scikit-learn==0.20.3
* jieba==0.39
* torch==0.4.1
* torchtext==0.3.1

### Dataset
We conduct our empirical experiments on two real-world legal datasets:
* **CAIL 2018**: contains criminal cases published by the Supreme People’s Court. Each case consists of two parts, i.e., fact description and corresponding judgment result (including laws, articles, and charges.
* **DPAM**: comprises 40,256 criminal cases. These data are crawled from China Judgment Online2 and span from Jan.2016 to June. 2016.

<!-- (https://arxiv.org/pdf/1807.02478.pdf) -->
<!-- (https://drive.google.com/open?id=1TCcTzte2cQ2wxWw-kXsZdUApCsNejdn8) -->

##### Example of Dataset
```json
{
  "text_len": 41,
  "laws": [234],
  "textIds": [2935,10,3,330,16,406,2935,1802,2,272,4328,1064,877,818,272,5455,....],
  "parent_class": ["侵犯公民人身"]
}
```

each instance contains four parts:
* **text_len**: the length of fact descriptions
* **parent class**: parent class
* **laws**: sub class
* **textIds**: the fact descriptions transformed from text to id

### Citation
For more information, please refer to our paper. If our work is helpful to you, please kindly cite our paper as:
```
  @inproceedings{HMN_SIGIR_2019
    author =  {PengFei Wang and Yu Fan. et.al.}
    title  =  {Hierarchical Matching Network for Crime Classification}
    booktitle = {SIGIR}
    year =  {2019}
  }
```

### Reference
* [CAIL2018: A Large-Scale Legal Dataset for Judgment Prediction](https://arxiv.org/pdf/1807.02478.pdf)
* [Modeling Dynamic Pairwise Attention for Crime Classification over Legal Articles](https://dl.acm.org/ft_gateway.cfm?id=3210057&type=pdf)

