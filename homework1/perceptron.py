import numpy as np
from data_process import get_dataset


class Perceptron:
    def __init__(self, w=None, b=None, w_shape=None):
        self.w = np.array(w) if w is not None else np.random.random(w_shape)
        self.b = b if b is not None else np.random.random(1)

    def train(self, train_X, train_Y, lr, max_iteration=10000):
        for _ in range(max_iteration):
            for x, y in zip(train_X, train_Y):
                if y * (self.w.dot(x.T) + self.b) <= 0:
                    self.w = self.w + lr * y * x
                    self.b = self.b + lr * y
                    break
            else:
                break

    def predict(self, x):
        return 1 if self.w.dot(x.T) + self.b > 0 else -1

    def accurate(self, X, Y):
        right_num = 0
        for i in range(X.shape[0]):
            if Y[i] == self.predict(X[i]):
                right_num += 1
        return right_num / X.shape[0]

    def get_args(self):
        print(f"w: {self.w}")
        print(f"b: {self.b}")


if __name__ == '__main__':
    train_set, test_set = get_dataset("./iris.data")
    perceptron = Perceptron(w_shape=4)
    perceptron.train(*train_set, 0.01)
    perceptron.get_args()
    print("训练集正确率：", end="")
    print(f"{perceptron.accurate(*train_set):.2%}")
    print("测试集正确率：", end="")
    print(f"{perceptron.accurate(*test_set):.2%}")





