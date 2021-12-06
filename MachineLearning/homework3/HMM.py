import numpy as np
from data_process import *
from typing import List


class HMM:
    def __init__(self, vocab_num, pos_num):
        # 词表中的单词数量
        self.vocab_num = vocab_num
        # 词性数量
        self.pos_num = pos_num
        self._init_transition_and_state_matrix()

    def _init_transition_and_state_matrix(self):
        # 状态转移矩阵, 记录词性 i -> j 的转移概率
        self.transition_matrix = np.zeros((self.pos_num, self.pos_num))
        # 观测状态矩阵, 记录单词 w 的词性为 i 的概率
        self.state_matrix = np.zeros((self.vocab_num, self.pos_num))
        # 初始观测状态, 记录词性 i 的概率
        self.start_state = np.zeros(self.pos_num)
        # 记录状态转移，用于计算转移矩阵
        self._transition = np.zeros((self.pos_num, self.pos_num))
        # 记录观测状态，用于计算状态矩阵和初始观测状态
        self._state = np.zeros((self.vocab_num, self.pos_num))

    # 计算转移矩阵、状态矩阵和初始状态
    def hmm(self, vocab_ids: List[List[int]], pos_labels: List[List[int]]):
        for sentence_ids, sentence_labels in zip(vocab_ids, pos_labels):
            for word_index in range(len(sentence_ids)):
                word_id = sentence_ids[word_index]
                pos_label = sentence_labels[word_index]
                # 记录状态加 1
                self._state[word_id][pos_label] += 1
                if word_index - 1 > 0:
                    last_pos_label = sentence_labels[word_index - 1]
                    # 转移状态加 1
                    self._transition[last_pos_label][pos_label] += 1
        # 计算转移概率矩阵、状态矩阵和初始状态向量
        np.divide(self._state, np.sum(self._state, axis=1, keepdims=True), out=self.state_matrix)
        self.state_matrix[np.isnan(self.state_matrix)] = 0

        np.divide(self._transition, np.sum(self._transition, axis=1, keepdims=True), out=self.transition_matrix)
        self.transition_matrix[np.isnan(self.transition_matrix)] = 0

        np.divide(np.sum(self._state, axis=0), np.sum(self._state), out=self.start_state)

    def vetebi(self, sentence_ids):
        word_num = len(sentence_ids)
        if word_num == 0:
            return []
        # 记录两个局部状态
        delta = np.zeros((word_num, self.pos_num))
        psi = np.zeros((word_num, self.pos_num), dtype=int)

        # 初始化两个局部状态
        np.multiply(self.start_state, self.state_matrix[sentence_ids[0]], out=delta[0])

        # 计算接下来的时间状态
        for word_index in range(1, word_num):
            word_id = sentence_ids[word_index]
            for pos_id in range(self.pos_num):
                temp = delta[word_index - 1] * self.transition_matrix[:, pos_id]
                max_index = temp.argmax()
                psi[word_index][pos_id] = max_index
                delta[word_index][pos_id] = temp[max_index] * self.state_matrix[word_id][pos_id]

        # 回溯
        pos_labels = []
        pos_label = delta[-1].argmax()
        pos_labels.append(pos_label)
        for step in range(word_num - 1, 0, -1):
            pos_label = psi[step][pos_label]
            pos_labels.append(pos_label)
        pos_labels.reverse()
        return pos_labels

    def acc(self, vocab_ids, golden_pos_labels):
        correct_pos_num = 0
        total_pos_num = 0
        for sentence_ids, golden_labels in zip(vocab_ids, golden_pos_labels):
            pred_labels = self.vetebi(sentence_ids)
            for pred_label, golden_label in zip(pred_labels, golden_labels):
                total_pos_num += 1
                if pred_label == golden_label:
                    correct_pos_num += 1
        return correct_pos_num / total_pos_num


if __name__ == '__main__':
    dataset = get_dataset("./corpus.txt")
    word_dict = dataset["word_dict"]
    train_set = dataset["train_set"]
    test_set = dataset["test_set"]

    hmm = HMM(len(word_dict), 49)
    hmm.hmm(*train_set)
    acc = hmm.acc(*test_set)
    print(f"acc: {acc:.2%}")
