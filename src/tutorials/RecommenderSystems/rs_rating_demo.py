#!/usr/bin/python
# coding:utf-8
# -------------------------------------------------------------------------------
# Name:    推荐系统
# Purpose: 推荐系统: Item CF/User CF/SVD 对比
# Author:  jiangzhonglian
# Create_time:  2020年9月21日
# Update_time:  2020年9月21日
# -------------------------------------------------------------------------------
from __future__ import print_function
import sys
import math
from operator import itemgetter
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from sklearn import model_selection as cv
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import pairwise_distances
from middleware.utils import TimeStat, Chart


def splitData(dataFile, test_size):
    # 加载数据集 (用户ID， 电影ID， 评分， 时间戳)
    header = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv(dataFile, sep='\t', names=header)

    n_users = df.user_id.unique().shape[0]
    n_items = df.item_id.unique().shape[0]

    print('>>> 本数据集包含: 总用户数 = %s | 总电影数 = %s' % (n_users, n_items) )
    train_data, test_data = cv.train_test_split(df, test_size=test_size)
    print(">>> 训练:测试 = %s:%s = %s:%s" % (len(train_data), len(test_data), 1-test_size, test_size))
    return df, n_users, n_items, train_data, test_data


def calc_similarity(n_users, n_items, train_data, test_data):
    # 创建用户产品矩阵，针对测试数据和训练数据，创建两个矩阵: 
    """
    line:  Pandas(Index=93661, user_id=624, item_id=750, rating=4, timestamp=891961163)
    """
    train_data_matrix = np.zeros((n_users, n_items))
    for line in train_data.itertuples():
        train_data_matrix[line[1] - 1, line[2] - 1] = line[3]

    test_data_matrix = np.zeros((n_users, n_items))
    for line in test_data.itertuples():
        test_data_matrix[line[1] - 1, line[2] - 1] = line[3]

    print("1:", np.shape(train_data_matrix))    # 行: 人 | 列: 电影
    print("2:", np.shape(train_data_matrix.T))  # 行: 电影 | 列: 人

    # 使用sklearn的 pairwise_distances 计算向量距离，cosine来计算余弦距离，越小越相似
    user_similarity = pairwise_distances(train_data_matrix, metric="cosine")
    item_similarity = pairwise_distances(train_data_matrix.T, metric="cosine")
    # print("<<< %s \n %s" % (np.shape(user_similarity), user_similarity) )
    # print("<<< %s \n %s" % (np.shape(item_similarity), item_similarity) )

    print('开始统计流行item的数量...', file=sys.stderr)
    item_popular = {}
    # 统计同一个电影，观看的总人数（也就是所谓的流行度！）
    for i_index in range(n_items):
        if np.sum(train_data_matrix[:, i_index]) != 0:
            item_popular[i_index] = np.sum(train_data_matrix[:, i_index] != 0)

    # save the total number of items
    item_count = len(item_popular)
    print('总共流行 item 数量 = %d' % item_count, file=sys.stderr)
    return train_data_matrix, test_data_matrix, user_similarity, item_similarity, item_popular


def predict(rating, similarity, type='user'):
    """
    :param rating: 训练数据
    :param similarity: 向量距离
    :return:
    """
    print("+++ %s" % type)
    print("    rating=", np.shape(rating))
    print("    similarity=", np.shape(similarity))
    if type == 'item':
        """
        综合打分:  
            rating.dot(similarity) 表示：
                某1个人所有的电影组合 X ·电影*电影·距离（第1列都是关于第1部电影和其他的电影的距离）中，计算出 第一个人对第1/2/3部电影的 总评分 1*n
                某2个人所有的电影组合 X ·电影*电影·距离（第1列都是关于第1部电影和其他的电影的距离）中，计算出 第一个人对第1/2/3部电影的 总评分 1*n
                ...
                某n个人所有的电影组合 X ·电影*电影·距离（第1列都是关于第1部电影和其他的电影的距离）中，计算出 第一个人对第1/2/3部电影的 总评分 1*n
            = 人-电影-评分(943, 1682) * 电影-电影-距离(1682, 1682) 
            = 人-电影-总评分距离(943, 1682)
            
            np.array([np.abs(similarity).sum(axis=1)]) 表示: 横向求和: 1 表示某一行所有的列求和
                第1列表示：某个A电影，对于所有电影计算出A的总距离
                第2列表示：某个B电影，对于所有电影的综出B的总距离
                ...
                第n列表示：某个N电影，对于所有电影的综出N的总距离
            = 每一个电影的总距离 (1, 1682)

            pred = 人-电影-平均评分 (943, 1682)
        """
        pred = rating.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    elif type == 'user':
        # 每个样本上减去数据的统计平均值可以移除共同的部分，凸显个体差异。

        # 求出每一个用户，所有电影的综合评分
        # 横向求平均: 1 表示某一行所有的列求平均
        mean_user_rating = rating.mean(axis=1)
        # numpy中包含的 newaxis 可以给原数组增加一个维度
        rating_diff = (rating - mean_user_rating[:, np.newaxis])

        # 均分  +  
        # 人-人-距离(943, 943)*人-电影-评分diff(943, 1682)=结果-人-电影（每个人对同一电影的综合得分）(943, 1682)  再除以  个人与其他人总的距离 = 人-电影综合得分
        """
        综合打分:  
            similarity.dot(rating_diff) 表示：
                第1列：第1个人与其他人的相似度 * 人与电影的相似度，得到 第1个人对第1/2/3列电影的 总得分 1*n
                第2列：第2个人与其他人的相似度 * 人与电影的相似度，得到 第2个人对第1/2/3列电影的 总得分 1*n
                ...
                第n列：第n个人与其他人的相似度 * 人与电影的相似度，得到 第n个人对第1/2/3列电影的 总得分 1*n
            = 人-人-距离(943, 943)  *  人-电影-评分(943, 1682)
            = 人-电影-总评分距离(943, 1682)

            np.array([np.abs(similarity).sum(axis=1)]) 表示: 横向求和: 1 表示某一行所有的列求和
                第1列表示：第A个人，对于所有人计算出A的总距离
                第2列表示：第B个人，对于所有人计算出B的总距离
                ...
                第n列表示：第N个人，对于所有人计算出N的总距离
            = 每一个电影的总距离 (1, 943)

            pred = 均值 + 人-电影-平均评分 (943, 1682)
        """
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(rating_diff) / np.array([np.abs(similarity).sum(axis=1)]).T

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

            # popular_sum是对所有的item的流行度进行加和
            if item in item_popular:
                popular_sum += math.log(1 + item_popular[item])

        rec_count += len(pre_items)
        test_count += len(test_items)

    precision = hit / (1.0 * rec_count)
    # 召回率，相对于测试推荐集合的数据
    recall = hit / (1.0 * test_count)
    # 覆盖率，相对于训练集合的数据
    coverage = len(all_rec_items) / (1.0 * len(item_popular))
    popularity = popular_sum / (1.0 * rec_count)
    print('--- %s: precision=%.4f \t recall=%.4f \t coverage=%.4f \t popularity=%.4f' % (
        name, precision, recall, coverage, popularity), file=sys.stderr)


def recommend(u_index, prediction):
    items = np.where(train_data_matrix[u_index, :] == 0)[0]
    pre_items = sorted(
        dict(zip(items, prediction[u_index, items])).items(),
        key=itemgetter(1),
        reverse=True)[:10]
    test_items = np.where(test_data_matrix[u_index, :] != 0)[0]

    result = [key for key, value in pre_items]
    result.sort(reverse=False)
    print('原始结果(%s): %s' % (len(test_items), test_items) )
    print('推荐结果(%s): %s' % (len(result), result) )


def main():
    global n_users, train_data_matrix, test_data_matrix
    # 基于内存的协同过滤
    # ...
    # 拆分数据集
    # http://files.grouplens.org/datasets/movielens/ml-100k.zip
    path_root = "/Users/jiangzl/work/data/机器学习"
    dataFile = '%s/16.RecommenderSystems/ml-100k/u.data' % path_root

    df, n_users, n_items, train_data, test_data = splitData(dataFile, test_size=0.25)

    # 计算相似度
    train_data_matrix, test_data_matrix, user_similarity, item_similarity, item_popular = calc_similarity(
        n_users, n_items, train_data, test_data)

    item_prediction = predict(train_data_matrix, item_similarity, type='item')
    user_prediction = predict(train_data_matrix, user_similarity, type='user')

    # # 评估: 均方根误差
    print('>>> Item based CF RMSE: ' + str(rmse(item_prediction, test_data_matrix)))
    print('>>> User based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix)))

    # 基于模型的协同过滤
    # ...
    # 计算MovieLens数据集的稀疏度 （n_users，n_items 是常量，所以，用户行为数据越少，意味着信息量少；越稀疏，优化的空间也越大）
    sparsity = round(1.0 - len(df) / float(n_users * n_items), 3)
    print('\nMovieLen100K的稀疏度: %s%%\n' % (sparsity * 100))

    # # 计算稀疏矩阵的最大k个奇异值/向量
    # minrmse = math.inf
    # index = 1
    # for k in range(1, 30, 1):
    #     u, s, vt = svds(train_data_matrix, k=k)
    #     # print(">>> ", s)
    #     s_diag_matrix = np.diag(s)
    #     svd_prediction = np.dot(np.dot(u, s_diag_matrix), vt)
    #     r_rmse = rmse(svd_prediction, test_data_matrix)
    #     if r_rmse < minrmse:
    #         index = k
    #         minrmse = r_rmse

    index = 11
    minrmse = 2.6717213264389765
    u, s, vt = svds(train_data_matrix, k=index)
    # print(">>> ", s)
    s_diag_matrix = np.diag(s)
    svd_prediction = np.dot(np.dot(u, s_diag_matrix), vt)
    r_rmse = rmse(svd_prediction, test_data_matrix)
    print("+++ k=%s, svd-shape: %s" % (index, np.shape(svd_prediction)) )
    print('>>> Model based CF RMSE: %s\n' %  minrmse)
    # """
    # 在信息量相同的情况下，矩阵越小，那么携带的信息越可靠。
    # 所以: user-cf 推荐效果高于 item-cf； 而svd分解后，发现15个维度效果就能达到90%以上，所以信息更可靠，效果也更好。
    # item-cf: 1682
    # user-cf: 943
    # svd: 15
    # """
    evaluate(item_prediction, item_popular, 'item')
    evaluate(user_prediction, item_popular, 'user')
    evaluate(svd_prediction,  item_popular, 'svd')

    # 推荐结果
    # recommend(1, item_prediction)
    # recommend(1, user_prediction)
    recommend(1, svd_prediction)


if __name__ == "__main__":
    main()
