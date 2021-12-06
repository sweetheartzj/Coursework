import re
import random


annotation_dict = {
    'a': 0, 'aq': 1, 'as': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'ga': 8, 'gn': 9, 'gv': 10, 'h': 11, 'i': 12,
        'ia': 13, 'ic': 14, 'in': 15, 'iv': 16, 'j': 17, "ja": 18, 'jn': 19, 'jv': 20, 'k': 21, 'm': 22, 'n': 23,
    'nd': 24,
        'ng': 25, 'nh': 26, 'ni': 27, 'nl': 28, 'nn': 29, 'ns': 30, 'nt': 31, 'nz': 32, 'o': 33, 'p': 34, 'q': 35,
        'r': 36, 'u': 37, 'v': 38, 'vd': 39, 'vi': 40, 'vl': 41, 'vt': 42, 'vu': 43, 'w': 44, 'wp': 45, 'ws': 46,
        'wu': 47, 'x': 48
    }


def get_dataset(file_path):
    sentences, annotations = get_sentence_and_annotation(file_path)
    word_dict = get_word_dict(sentences)
    sentence_ids = [[convert_word_to_id(word, word_dict) for word in sentence] for sentence in sentences]
    label = [[convert_anno_to_label(anno, annotation_dict) for anno in annotation] for annotation in annotations]
    train_set, test_set = split_dataset(sentence_ids, label)
    return {"word_dict": word_dict, "train_set": train_set, "test_set": test_set}


# 得到每条数据的单词和对应的标注
def get_sentence_and_annotation(file_path):
    sentences = []
    annotations = []
    with open(file_path, "r", encoding="utf8") as data_file:
        while data_file.readable() and (line := data_file.readline()):
            match = re.match(r"\d+[\t ]+(.+)", line)
            if match is not None:
                content = match.group(1)
                sentence, annotation = split_word_and_annotation(content)
                sentences.append(sentence)
                annotations.append(annotation)
    return sentences, annotations


# 分离单条数据的单词和标注
def split_word_and_annotation(content: str):
    words = []
    annotations = []
    word_and_annotations = re.findall(r"(.+?)/([a-z]+) *", content)
    for word, annotation in word_and_annotations:
        words.append(word.strip())
        annotations.append(annotation)
    return words, annotations


# 建立单词词表
def get_word_dict(sentences):
    words = set()
    word_dict = {}
    Id = 0
    for sentence in sentences:
        for word in sentence:
            if word not in words:
                words.add(word)
                word_dict[word] = Id
                Id += 1
    return word_dict


# 这里不考虑单词不在词表中的情形
def convert_word_to_id(word, word_dict):
    return word_dict[word]


def convert_anno_to_label(annotation, anno_dict):
    return anno_dict[annotation]


def split_dataset(sentences, annotations, train_weight=9, test_weight=1):
    seed = random.randint(0, 100)
    random.seed(seed)
    random.shuffle(sentences)
    random.seed(seed)
    random.shuffle(annotations)
    num = len(sentences)
    train_num = int(num * train_weight / (train_weight + test_weight))
    train_set = sentences[: train_num], annotations[: train_num]
    test_set = sentences[train_num:], annotations[train_num:]
    return train_set, test_set



