import numpy as np
from data_process import get_dataset


class SVM:
    def __init__(self, C=1, kernel="linear"):
        self.C = C
        self.kernel = kernel

    def _kernel_function(self, x1, x2):
        if self.kernel == 'linear':
            return np.dot(x1, x2.T)
        elif self.kernel == 'poly':
            return np.dot(x1, x2.T) ** 2
        return 0

    def _init_args(self, train_X, train_Y):
        self.train_X = train_X
        self.train_Y = train_Y
        self.alphas = np.random.random(train_Y.shape)
        self.b = np.random.random(1)
        self.data_size = train_X.shape[0]
        self._E = [self._get_e(i) for i in range(self.data_size)]

    def _get_g(self, xi):
        """
        计算 g(xi)
        :param xi:
        :return:
        """
        w = np.multiply(self.alphas, self.train_Y)
        kernel_result = self._kernel_function(xi, self.train_X)
        return np.dot(kernel_result, w) + self.b

    def _get_e(self, i):
        """
        计算误差 E
        :param x:
        :param y:
        :return:
        """
        x_i = self.train_X[i]
        y_i = self.train_Y[i]
        return self._get_g(x_i) - y_i

    def _choose_alpha(self):
        """
        选择 alpha_i 和 alpha_j
        alpha_i 是违背 KTT 条件的变量，外循环
        alpha_j 是与 alpha_i 对应偏差最大的变量，内循环
        :return: i 和 j
        """
        support_vector_index = []
        un_support_vector_index = []
        for index in range(self.data_size):
            if 0 <= self.alphas[index] <= self.C:
                support_vector_index.append(index)
            else:
                un_support_vector_index.append(index)
        # 先看可能是支持向量的 alpha_i
        index_i = support_vector_index + un_support_vector_index
        for i in index_i:
            if self._check_KKT(i):
                continue
            else:
                E1 = self._E[i]
                # 选择变化最大 的 alpha_j
                if E1 >= 0:
                    j = min(range(self.data_size), key=lambda x: self._E[x])
                else:
                    j = max(range(self.data_size), key=lambda x: self._E[x])
                return i, j
        # 返回 None 说明所有的参数都符合 KTT 条件
        return None

    def _check_KKT(self, i):
        """
        判断 alpha_i 是否符合 KKT 条件
        :param i:
        :return:
        """
        alpha_i = self.alphas[i]
        x_i = self.train_X[i]
        y_i = self.train_Y[i]
        g_x_i = self._get_g(x_i)
        if alpha_i == 0:
            return y_i * g_x_i >= 1
        elif 0 < alpha_i < self.C:
            return y_i * g_x_i == 1
        else:
            return y_i * g_x_i <= 1

    def train(self, train_X, train_Y, max_iteration=50000):
        self._init_args(train_X, train_Y)
        self._SMO(max_iteration)

    def _SMO(self, max_iteration=1000000):
        """
        SMO 算法
        :param max_iteration: 最大迭代次数
        """
        for iteration in range(max_iteration):
            # 1. 选取 alpha1 和 alpha2
            if (alpha := self._choose_alpha()) is not None:
                i, j = alpha
            else:
                break
            # 计算相关值
            alpha1, alpha2 = self.alphas[i], self.alphas[j]
            x1, x2 = self.train_X[i], self.train_X[j]
            y1, y2 = self.train_Y[i], self.train_Y[j]

            # 2. 更新 alpha2
            E1, E2 = self._E[i], self._E[j]
            eta = self._kernel_function(self.train_X[i, :], self.train_X[i, :]) + self._kernel_function(self.train_X[j, :], self.train_X[j, :])
            - 2 * self._kernel_function(self.train_X[i, :], self.train_X[j, :])
            alpha2_new_unc = alpha2 + y2 * (E1 - E2) / eta
            # 3. 剪枝 alpha2
            if y1 == y2:
                L = max(0, alpha1 + alpha2 - self.C)
                H = min(self.C, alpha1 + alpha2)
            else:
                L = max(0, alpha2 - alpha1)
                H = min(self.C, self.C + alpha2 - alpha1)
            if alpha2_new_unc > H:
                alpha2_new = H
            elif alpha2_new_unc < L:
                alpha2_new = L
            else:
                alpha2_new = alpha2_new_unc
            # 4. 更新 alpha1
            alpha1_new = alpha1 + y1 * y2 * (alpha2 - alpha2_new)
            # 5. 更新 b 和 E 值
            b1_new = - E1 - y1 * self._kernel_function(x1, x1) * (alpha1_new - alpha1) \
                     - y2 * self._kernel_function(x2, x1) * (alpha2_new - alpha2) + self.b
            b2_new = - E2 - y1 * self._kernel_function(x1, x2) * (alpha1_new - alpha1) \
                     - y2 * self._kernel_function(x2, x2) * (alpha2_new - alpha2) + self.b
            if 0 < alpha1_new < self.C:
                b_new = b1_new
            elif 0 < alpha2_new < self.C:
                b_new = b2_new
            else:
                b_new = (b1_new + b2_new) / 2
            self.b = b_new
            self._E[i] = self._get_e(i)
            self._E[j] = self._get_e(j)
            self.alphas[i] = alpha1_new
            self.alphas[j] = alpha2_new

    def predict(self, x):
        """
        预测
        :param x: 1*d 维的特征值
        :return: 预测结果
        """
        return 1 if self._get_g(x) > 0 else -1

    def accurate(self, X, Y):
        """
        计算准确度
        :param X: 维度是(M*d)的特征值
        :return:
        """
        right_num = 0
        for i in range(X.shape[0]):
            if Y[i] == self.predict(X[i]):
                right_num += 1
        return right_num / X.shape[0]


if __name__ == '__main__':
    train_set, test_set = get_dataset("./iris.data", 9, 1)
    svm = SVM()
    svm.train(*train_set, max_iteration=1000)
    print(svm.accurate(*train_set))
    print(svm.accurate(*test_set))
    # print(svm.alphas)






