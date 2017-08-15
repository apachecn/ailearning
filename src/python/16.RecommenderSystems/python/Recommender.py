import numpy as np


# 自定义杰卡德相似系数函数，仅对0-1矩阵有效
def Jaccard(a, b):
    return 1.0*(a*b).sum()/(a+b-a*b).sum()


class Recommender():

    # 相似度矩阵
    sim = None

    # 计算相似度矩阵的函数
    def similarity(self, x, distance):
        y = np.ones((len(x), len(x)))
        for i in range(len(x)):
            for j in range(len(x)):
                y[i, j] = distance(x[i], x[j])
        return y

    # 训练函数
    def fit(self, x, distance=Jaccard):
        self.sim = self.similarity(x, distance)

    # 推荐函数
    def recommend(self, a):
        return np.dot(self.sim, a)*(1-a)
