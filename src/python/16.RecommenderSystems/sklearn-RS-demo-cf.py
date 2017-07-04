#!/usr/bin/python
# coding:utf8

from math import sqrt

import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from sklearn import cross_validation as cv
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import pairwise_distances

# 加载数据集
header = ['user_id', 'item_id', 'rating', 'timestamp']
# http://files.grouplens.org/datasets/movielens/ml-100k.zip
dataFile = 'input/16.RecommenderSystems/ml-100k/u.data'
df = pd.read_csv(dataFile, sep='\t', names=header)

n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]
print 'Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items)

# 拆分数据集
train_data, test_data = cv.train_test_split(df, test_size=0.25)

# 创建用户产品矩阵，针对测试数据和训练数据，创建两个矩阵：
train_data_matrix = np.zeros((n_users, n_items))
for line in train_data.itertuples():
    train_data_matrix[line[1]-1, line[2]-1] = line[3]
test_data_matrix = np.zeros((n_users, n_items))
for line in test_data.itertuples():
    test_data_matrix[line[1]-1, line[2]-1] = line[3]
# 使用sklearn的pairwise_distances函数来计算余弦相似性。
user_similarity = pairwise_distances(train_data_matrix, metric="cosine")
item_similarity = pairwise_distances(train_data_matrix.T, metric="cosine")


def predict(rating, similarity, type='user'):
    if type == 'user':
        mean_user_rating = rating.mean(axis=1)
        rating_diff = (rating - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(rating_diff)/np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = rating.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred


user_prediction = predict(train_data_matrix, user_similarity, type='user')
item_prediction = predict(train_data_matrix, item_similarity, type='item')


def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))


print 'User based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix))
print 'Item based CF RMSe: ' + str(rmse(item_prediction, test_data_matrix))

sparsity = round(1.0 - len(df)/float(n_users*n_items), 3)
print 'The sparsity level of MovieLen100K is ' + str(sparsity * 100) + '%'


u, s, vt = svds(train_data_matrix, k=20)
s_diag_matrix = np.diag(s)
x_pred = np.dot(np.dot(u, s_diag_matrix), vt)
print 'User-based CF MSE: ' + str(rmse(x_pred, test_data_matrix))
