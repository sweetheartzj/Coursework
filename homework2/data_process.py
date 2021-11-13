import re
import random


def process_file(data_path="./AsianReligionsData.txt"):
    # 文本
    texts = []
    # 文本所对应的书籍
    labels = []
    with open(data_path, "r") as data_file:
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
    num = len(texts)












