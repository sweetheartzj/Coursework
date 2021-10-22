import random
import numpy as np


def read_file(file_path):
    """
    读取文件
    :param file_path: 文件路径
    :return: 数据和标签
    """
    X = []
    Y = []
    with open(file_path, "r", encoding="utf8") as file:
        while file.readable() and (line := file.readline()):
            one_iris_data = line.strip().split(",")
            x = list(map(lambda i: float(i), one_iris_data[: -1]))
            # Iris-setosa 标签为 1， Iris-versicolor 标签为 -1
            y = 1 if one_iris_data[-1] == "Iris-setosa" else -1
            X.append(x)
            Y.append(y)
    return X, Y


def shuffle_data(X, Y):
    """
    打乱数据集
    :param X: 数据列表
    :param Y: 对应的标签列表
    :return: 打乱后的数据集
    """
    # 使用相同的随机种子，确保打乱的顺序一致
    seed = random.randint(0, 100)
    random.seed(seed)
    random.shuffle(X)
    random.seed(seed)
    random.shuffle(Y)


def get_dataset(file_path, train_ratio=9, test_ratio=1):
    """
    封装函数，获取训练集和测试集
    :param file_path: 文件路径
    :param train_ratio: 训练集的比例
    :param test_ratio: 测试集的比例
    :return: 测试机和训练集
    """
    X, Y = read_file(file_path)
    shuffle_data(X, Y)
    train_num = int(len(X) / (train_ratio + test_ratio) * train_ratio)
    train_set = np.array(X[: train_num]), np.array(Y[: train_num])
    test_set = np.array(X[train_num:]), np.array(Y[train_num:])
    return train_set, test_set



