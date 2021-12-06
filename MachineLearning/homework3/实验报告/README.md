# HMM

目录结构：

```
HMM
├── corpus.txt
├── data_process.py
├── HMM.py
```
* `corpus.txt`: 标注数据文件
* `data_process.py`: 数据处理
* `HMM.py`: HMM 实现

`data_process.py`:

* `get_dataset(file_path):`
  * 主函数，输入文件地址，返回词典，训练集和测试集

* `get_sentence_and_annotation(file_path):`
  * 辅助函数，输入文件地址，返回序列集和对应的词性标注集

* `split_word_and_annotation(content):`
  * 辅助函数，输入一条数据，分离该条数据的序列和词性

* `get_word_dict(sentences):`
  * 辅助函数，输入序列集，建立词表并返回


`HMM.py:`

```python
class HMM:
    def __init__(self, vocab_num, pos_num):
        ...
        return
    
    def hmm(self, vocab_ids, pos_labels):
        ...
        return
    
    def vetebi(self, sentence_ids):
        ...
        retun
    def acc(self, vocab_ids, golden_pos_labels):
        ...
        return
```

* `__init__(self, vocab_num, pos_num):`
  * 输入词表大小和词性集大小，初始化转移矩阵、状态矩阵和初始状态
* `hmm(self, vocab_ids, pos_labels)`:
  * 输入序列集和对应的词性集，计算转移矩阵、状态矩阵和初始状态
* `vetebi(self, sentence_ids):`
  * 预测函数，输入序列，利用维特比算法输出预测的词性’
* `acc(self, vocab_ids, golden_pos_labels):`
  * 评价函数，计算正确率