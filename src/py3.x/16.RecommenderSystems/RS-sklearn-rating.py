#!/usr/bin/python
# coding:utf8

import sys
import math
from operator import itemgetter

import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from sklearn import cross_validation as cv
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import pairwise_distances


def splitData(dataFile, test_size):
    # 加载数据集
    header = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv(dataFile, sep='\t', names=header)

    n_users = df.user_id.unique().shape[0]
    n_items = df.item_id.unique().shape[0]

    print('Number of users = ' + str(n_users) + ' | Number of movies = ' +
          str(n_items))
    train_data, test_data = cv.train_test_split(df, test_size=test_size)
    print("数据量：", len(train_data), len(test_data))
    return df, n_users, n_items, train_data, test_data


def calc_similarity(n_users, n_items, train_data, test_data):
    # 创建用户产品矩阵，针对测试数据和训练数据，创建两个矩阵：
    train_data_matrix = np.zeros((n_users, n_items))
    for line in train_data.itertuples():
        train_data_matrix[line[1] - 1, line[2] - 1] = line[3]
    test_data_matrix = np.zeros((n_users, n_items))
    for line in test_data.itertuples():
        test_data_matrix[line[1] - 1, line[2] - 1] = line[3]

    # 使用sklearn的pairwise_distances函数来计算余弦相似性。
    print("1:", np.shape(train_data_matrix))  # 行：人，列：电影
    print("2:", np.shape(train_data_matrix.T))  # 行：电影，列：人

    user_similarity = pairwise_distances(train_data_matrix, metric="cosine")
    item_similarity = pairwise_distances(train_data_matrix.T, metric="cosine")

    print >> sys.stderr, '开始统计流行item的数量...'
    item_popular = {}
    # 统计在所有的用户中，不同电影的总出现次数
    for i_index in range(n_items):
        if np.sum(train_data_matrix[:, i_index]) != 0:
            item_popular[i_index] = np.sum(train_data_matrix[:, i_index] != 0)
            # print "pop=", i_index, self.item_popular[i_index]

    # save the total number of items
    item_count = len(item_popular)
    print >> sys.stderr, '总共流行item数量 = %d' % item_count

    return train_data_matrix, test_data_matrix, user_similarity, item_similarity, item_popular


def predict(rating, similarity, type='user'):
    print(type)
    print("rating=", np.shape(rating))
    print("similarity=", np.shape(similarity))
    if type == 'user':
        # 求出每一个用户，所有电影的综合评分（axis=0 表示对列操作， 1表示对行操作）
        # print "rating=", np.shape(rating)
        mean_user_rating = rating.mean(axis=1)
        # np.newaxis参考地址: http://blog.csdn.net/xtingjie/article/details/72510834
        # print "mean_user_rating=", np.shape(mean_user_rating)
        # print "mean_user_rating.newaxis=", np.shape(mean_user_rating[:, np.newaxis])
        rating_diff = (rating - mean_user_rating[:, np.newaxis])
        # print "rating=", rating[:3, :3]
        # print "mean_user_rating[:, np.newaxis]=", mean_user_rating[:, np.newaxis][:3, :3]
        # print "rating_diff=", rating_diff[:3, :3]

        # 均分  +  人-人-距离(943, 943)*人-电影-评分diff(943, 1682)=结果-人-电影（每个人对同一电影的综合得分）(943, 1682)  再除以  个人与其他人总的距离 = 人-电影综合得分
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(
            rating_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        # 综合打分： 人-电影-评分(943, 1682)*电影-电影-距离(1682, 1682)=结果-人-电影(各个电影对同一电影的综合得分)(943, 1682)  ／  再除以  电影与其他电影总的距离 = 人-电影综合得分
        pred = rating.dot(similarity) / np.array(
            [np.abs(similarity).sum(axis=1)])
    return pred


def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return math.sqrt(mean_squared_error(prediction, ground_truth))


def evaluate(prediction, item_popular, name):
    hit = 0
    rec_count = 0
    test_count = 0
    popular_sum = 0
    all_rec_items = set()
    for u_index in range(n_users):
        items = np.where(train_data_matrix[u_index, :] == 0)[0]
        pre_items = sorted(
            dict(zip(items, prediction[u_index, items])).items(),
            key=itemgetter(1),
            reverse=True)[:20]
        test_items = np.where(test_data_matrix[u_index, :] != 0)[0]

        # 对比测试集和推荐集的差异 item, w
        for item, _ in pre_items:
            if item in test_items:
                hit += 1
            all_rec_items.add(item)

            # 计算用户对应的电影出现次数log值的sum加和
            if item in item_popular:
                popular_sum += math.log(1 + item_popular[item])

        rec_count += len(pre_items)
        test_count += len(test_items)

    precision = hit / (1.0 * rec_count)
    recall = hit / (1.0 * test_count)
    coverage = len(all_rec_items) / (1.0 * len(item_popular))
    popularity = popular_sum / (1.0 * rec_count)
    print >> sys.stderr, '%s: precision=%.4f \t recall=%.4f \t coverage=%.4f \t popularity=%.4f' % (
        name, precision, recall, coverage, popularity)


def recommend(u_index, prediction):
    items = np.where(train_data_matrix[u_index, :] == 0)[0]
    pre_items = sorted(
        dict(zip(items, prediction[u_index, items])).items(),
        key=itemgetter(1),
        reverse=True)[:10]
    test_items = np.where(test_data_matrix[u_index, :] != 0)[0]

    print('原始结果：', test_items)
    print('推荐结果：', [key for key, value in pre_items])


if __name__ == "__main__":

    # 基于内存的协同过滤
    # ...
    # 拆分数据集
    # http://files.grouplens.org/datasets/movielens/ml-100k.zip
    dataFile = 'db/16.RecommenderSystems/ml-100k/u.data'
    df, n_users, n_items, train_data, test_data = splitData(
        dataFile, test_size=0.25)

    # 计算相似度
    train_data_matrix, test_data_matrix, user_similarity, item_similarity, item_popular = calc_similarity(
        n_users, n_items, train_data, test_data)

    item_prediction = predict(train_data_matrix, item_similarity, type='item')
    user_prediction = predict(train_data_matrix, user_similarity, type='user')

    # 评估：均方根误差
    print(
        'Item based CF RMSE: ' + str(rmse(item_prediction, test_data_matrix)))
    print(
        'User based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix)))

    # 基于模型的协同过滤
    # ...
    # 计算MovieLens数据集的稀疏度 （n_users，n_items 是常量，所以，用户行为数据越少，意味着信息量少；越稀疏，优化的空间也越大）
    sparsity = round(1.0 - len(df) / float(n_users * n_items), 3)
    print('The sparsity level of MovieLen100K is ' + str(sparsity * 100) + '%')

    # 计算稀疏矩阵的最大k个奇异值/向量
    u, s, vt = svds(train_data_matrix, k=15)
    s_diag_matrix = np.diag(s)
    svd_prediction = np.dot(np.dot(u, s_diag_matrix), vt)
    print("svd-shape:", np.shape(svd_prediction))
    print(
        'Model based CF RMSE: ' + str(rmse(svd_prediction, test_data_matrix)))
    """
    在信息量相同的情况下，矩阵越小，那么携带的信息越可靠。
    所以：user-cf 推荐效果高于 item-cf； 而svd分解后，发现15个维度效果就能达到90%以上，所以信息更可靠，效果也更好。
    item-cf: 1682
    user-cf: 943
    svd: 15
    """
    evaluate(item_prediction, item_popular, 'item')
    evaluate(user_prediction, item_popular, 'user')
    evaluate(svd_prediction, item_popular, 'svd')

    # 推荐结果
    recommend(1, svd_prediction)
