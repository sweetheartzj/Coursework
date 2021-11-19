import numpy as np
from data_process import get_dataset


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LogisticRegression:
    def __init__(self, w=None, b=None, w_shape=None):
        self.w = w if w is not None else np.random.random(w_shape)
        self.b = b if b is not None else np.random.random()

    # 梯度下降法
    def train(self, X, Y, lr=0.01, max_iteration=1000):
        for iteration in range(max_iteration):
            result = sigmoid(np.dot(self.w, X.T) + self.b)
            error = result - Y
            grad = np.dot(X.T, error)
            self.w = self.w - lr * grad
            self.b = self.b - lr * np.sum(error)

    def predict(self, x):
        return 1 if np.dot(self.w, x.T) + self.b >= 0 else 0

    def accurate(self, X, Y):
        linear_out = np.dot(self.w, X.T)
        correct_num = np.sum((linear_out > 0) == Y)
        return correct_num / X.shape[0]

    def get_args(self):
        print(f"w: {self.w}")
        print(f"b: {self.b}")


if __name__ == '__main__':
    train_set, test_set = get_dataset("./iris.data", 9, 1, label_class=1)
    logistic = LogisticRegression(w_shape=4)
    logistic.train(*train_set)
    logistic.get_args()
    print("训练集正确率：", end="")
    print(logistic.accurate(*train_set))
    print("测试集正确率：", end="")
    print(logistic.accurate(*test_set))




