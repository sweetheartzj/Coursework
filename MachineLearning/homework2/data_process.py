import re
import random


def process_file(data_path="./AsianReligionsData.txt"):
    # 文本
    texts = []
    # 文本所对应的书籍
    labels = []
    with open(data_path, "r", encoding="utf8") as data_file:
        current_book = 0
        # 匹配书籍和章节
        pattern = re.compile(r"(\d+)\.\d+")
        while data_file.readable() and (line := data_file.readline()):
            if (book := pattern.match(line)) is not None:
                current_book = int(book.group(1))
                continue
            # 去除文本中的章节符号
            texts.append(re.sub(r"§ \d+\. ?", "", line))
            labels.append(current_book)
    return texts, labels


def split_dataset(texts, labels, train_num=9, test_num=1):
    train_X = []
    train_Y = []
    test_X = []
    test_Y = []
    num = len(texts)
    train_num = int(num * train_num / (train_num + test_num))
    train_index = random.sample(range(num), train_num)
    for index in range(num):
        if index in train_index:
            train_X.append(texts[index])
            train_Y.append(labels[index])
        else:
            test_X.append(texts[index])
            test_Y.append(labels[index])
    return (train_X, train_Y), (test_X, test_Y)


def get_dataset(data_path, train_num=9, test_num=1):
    texts, labels = process_file(data_path)
    train_set, test_set = split_dataset(texts, labels, train_num, test_num)
    return train_set, test_set











