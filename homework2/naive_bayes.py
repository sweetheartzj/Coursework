from data_process import get_dataset
import re
import string
from typing import List


class NaiveBayesClassifier:
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        # 记录书本信息，键为书籍标签，值为出现的数量
        self.books = {}
        # 记录单词信息，[{"book_label": 0, "word_num": 100, "words": {}]
        self.statistics = []
        # 单词可能的取值
        self.words = set()
        self.train()

    # 记录训练集每本书中每个单词出现的频率
    def train(self):
        for text, label in zip(self.texts, self.labels):
            words = re.sub(re.escape(string.punctuation), " ", text).split()
            self.books[label] = self.books.get(label, 0) + 1
            book_info = self.get_book_info(label)
            if book_info is None:
                book_info = {"book_label": label, "word_num": 0, "words": {}}
                self.statistics.append(book_info)
            book_info["word_num"] += len(words)
            for word in words:
                self.words.add(word)
                book_info["words"][word] = book_info["words"].get(word, 0) + 1

    def get_book_info(self, book_label):
        for book_info in self.statistics:
            if book_info.get("book_label") == book_label:
                return book_info

    def get_probability_of_book(self, book_label, laplace=0):
        # 该书籍的数量
        book_num = self.books.get(book_label, 0)
        # 总共的书籍数量
        total_book_num = sum(self.books.values())
        # 书籍的种类数量
        book_type_num = sum(1 for _ in self.books)
        return (book_num + laplace) / (total_book_num + book_type_num * laplace)

    def predict(self, text, laplace=0):
        words = re.sub(re.escape(string.punctuation), " ", text).split()
        probabilities = []
        for book in self.books:
            book_info = self.get_book_info(book)
            # 计算 P(c_i)
            p_book = self.get_probability_of_book(book, laplace=laplace)
            probability = p_book
            for word in words:
                # 该书籍中该单词的数量
                word_num = book_info["words"].get(word, 0)
                # 该书籍总的单词数量
                total_word_num = book_info.get("word_num", 0)
                # 单词的取值数量
                word_type_num = len(self.words)
                # 计算 P(w_j|c_i)
                p = (word_num + laplace) / (total_word_num + word_type_num * laplace)
                probability *= p
            probabilities.append((book, probability))
        return max(probabilities, key=lambda x: x[1])[0]

    def accurate(self, texts, labels, laplace=0):
        pred_labels = [self.predict(text, laplace=laplace) for text in texts]
        correct_num = 0
        total_num = 0
        for pred_label, golden_label in zip(pred_labels, labels):
            total_num += 1
            if pred_label == golden_label:
                correct_num += 1
        return correct_num / total_num


if __name__ == '__main__':
    train_set, test_set = get_dataset("./AsianReligionsData.txt")
    classifier = NaiveBayesClassifier(*train_set)
    print(f"lambda: 0, 正确率为: {classifier.accurate(*test_set, laplace=0): .2%}")
    print(f"lambda: 0.5, 正确率为: {classifier.accurate(*test_set, laplace=0.5): .2%}")
    print(f"lambda: 1, 正确率为: {classifier.accurate(*test_set, laplace=1): .2%}")
    print(f"lambda: 1.5, 正确率为: {classifier.accurate(*test_set, laplace=1.5): .2%}")
    print(f"lambda: 2, 正确率为: {classifier.accurate(*test_set, laplace=2): .2%}")











