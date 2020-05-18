#!/usr/bin/python
# coding:utf8
'''
Created on 2015-06-22
Update  on 2017-05-16
Author: Lockvictor/片刻
《推荐系统实践》协同过滤算法源代码
参考地址: https://github.com/Lockvictor/MovieLens-RecSys
更新地址: https://github.com/apachecn/AiLearning
'''
from __future__ import print_function
import math
import random
import sys
from operator import itemgetter

import numpy as np
import pandas as pd
from sklearn import cross_validation as cv
from sklearn.metrics.pairwise import pairwise_distances

# 作用: 使得随机数据可预测
random.seed(0)


class ItemBasedCF():
    ''' TopN recommendation - ItemBasedCF '''

    def __init__(self):
        # 拆分数据集
        self.train_mat = {}
        self.test_mat = {}

        # 总用户数
        self.n_users = 0
        self.n_items = 0

        # n_sim_user: top 20个用户， n_rec_item: top 10个推荐结果
        self.n_sim_item = 20
        self.n_rec_item = 10

        # item_mat_similarity: 电影之间的相似度， item_popular: 电影的出现次数， item_count: 总电影数量
        self.item_mat_similarity = {}
        self.item_popular = {}
        self.item_count = 0

        print('Similar item number = %d' % self.n_sim_item, file=sys.stderr)
        print('Recommended item number = %d' % self.n_rec_item, file=sys.stderr)

    def splitData(self, dataFile, test_size):
        # 加载数据集
        header = ['user_id', 'item_id', 'rating', 'timestamp']
        df = pd.read_csv(dataFile, sep='\t', names=header)

        self.n_users = df.user_id.unique().shape[0]
        self.n_items = df.item_id.unique().shape[0]

        print('Number of users = ' + str(self.n_users) +
              ' | Number of items = ' + str(self.n_items))

        # 拆分数据集:  用户+电影
        self.train_data, self.test_data = cv.train_test_split(
            df, test_size=test_size)
        print('分离训练集和测试集成功', file=sys.stderr)
        print('len(train) = %s' % np.shape(self.train_data)[0], file=sys.stderr)
        print('len(test) = %s' % np.shape(self.test_data)[0], file=sys.stderr)

    def calc_similarity(self):
        # 创建用户产品矩阵，针对测试数据和训练数据，创建两个矩阵: 
        self.train_mat = np.zeros((self.n_users, self.n_items))
        for line in self.train_data.itertuples():
            self.train_mat[int(line.user_id) - 1,
                           int(line.item_id) - 1] = float(line.rating)
        self.test_mat = np.zeros((self.n_users, self.n_items))
        for line in self.test_data.itertuples():
            # print "line", line.user_id-1, line.item_id-1, line.rating
            self.test_mat[int(line.user_id) - 1,
                          int(line.item_id) - 1] = float(line.rating)

        # 使用sklearn的pairwise_distances函数来计算余弦相似性。
        print("1:", np.shape(np.mat(self.train_mat).T))  # 行: 电影，列: 人
        # 电影-电影-距离(1682, 1682)
        self.item_mat_similarity = pairwise_distances(
            np.mat(self.train_mat).T, metric='cosine')
        print('item_mat_similarity=', np.shape(
            self.item_mat_similarity), file=sys.stderr)

        print('开始统计流行item的数量...', file=sys.stderr)

        # 统计在所有的用户中，不同电影的总出现次数
        for i_index in range(self.n_items):
            if np.sum(self.train_mat[:, i_index]) != 0:
                self.item_popular[i_index] = np.sum(
                    self.train_mat[:, i_index] != 0)
                # print "pop=", i_index, self.item_popular[i_index]

        # save the total number of items
        self.item_count = len(self.item_popular)
        print('总共流行item数量 = %d' % self.item_count, file=sys.stderr)

    # @profile
    def recommend(self, u_index):
        """recommend(找出top K的电影，对电影进行相似度sum的排序，取出top N的电影数)

        Args:
            u_index   用户_ID-1=用户index
        Returns:
            rec_item  电影推荐列表，按照相似度从大到小的排序
        """
        ''' Find K similar items and recommend N items. '''
        K = self.n_sim_item
        N = self.n_rec_item
        rank = {}
        i_items = np.where(self.train_mat[u_index, :] != 0)[0]
        # print "i_items=", i_items
        watched_items = dict(zip(i_items, self.train_mat[u_index, i_items]))

        # 计算top K 电影的相似度
        # rating=电影评分, w=不同电影出现的次数
        # 耗时分析: 98.2%的时间在 line-154行
        for i_item, rating in watched_items.items():
            i_other_items = np.where(
                self.item_mat_similarity[i_item, :] != 0)[0]
            for related_item, w in sorted(
                    dict(
                        zip(i_other_items, self.item_mat_similarity[
                            i_item, i_other_items])).items(),
                    key=itemgetter(1),
                    reverse=True)[0:K]:
                if related_item in watched_items:
                    continue
                rank.setdefault(related_item, 0)
                rank[related_item] += w * rating

        # return the N best items
        return sorted(rank.items(), key=itemgetter(1), reverse=True)[0:N]

    def evaluate(self):
        ''' return precision, recall, coverage and popularity '''
        print('Evaluation start...', file=sys.stderr)

        # varables for precision and recall
        # hit表示命中(测试集和推荐集相同+1)，rec_count 每个用户的推荐数， test_count 每个用户对应的测试数据集的电影数
        hit = 0
        rec_count = 0
        test_count = 0
        # varables for coverage
        all_rec_items = set()
        # varables for popularity
        popular_sum = 0

        # enumerate 将其组成一个索引序列，利用它可以同时获得索引和值
        # 参考地址: http://blog.csdn.net/churximi/article/details/51648388
        for u_index in range(50):
            if u_index > 0 and u_index % 10 == 0:
                print('recommended for %d users' % u_index, file=sys.stderr)
            print("u_index", u_index)

            # 对比测试集和推荐集的差异
            rec_items = self.recommend(u_index)
            print("rec_items=", rec_items)
            # item, w
            for item, _ in rec_items:
                # print 'test_mat[u_index, item]=', item, self.test_mat[u_index, item]

                if self.test_mat[u_index, item] != 0:
                    hit += 1
                    print("self.test_mat[%d, %d]=%s" %
                          (u_index, item, self.test_mat[u_index, item]))
                # 计算用户对应的电影出现次数log值的sum加和
                if item in self.item_popular:
                    popular_sum += math.log(1 + self.item_popular[item])

            rec_count += len(rec_items)
            test_count += np.sum(self.test_mat[u_index, :] != 0)
            # print "test_count=", np.sum(self.test_mat[u_index, :] != 0), np.sum(self.train_mat[u_index, :] != 0)

        print("-------", hit, rec_count)
        precision = hit / (1.0 * rec_count)
        recall = hit / (1.0 * test_count)
        coverage = len(all_rec_items) / (1.0 * self.item_count)
        popularity = popular_sum / (1.0 * rec_count)

        print('precision=%.4f \t recall=%.4f \t coverage=%.4f \t popularity=%.4f' % (
            precision, recall, coverage, popularity), file=sys.stderr)


if __name__ == '__main__':
    dataFile = 'data/16.RecommenderSystems/ml-100k/u.data'

    # 创建ItemCF对象
    itemcf = ItemBasedCF()
    # 将数据按照 7:3的比例，拆分成: 训练集和测试集，存储在usercf的trainset和testset中
    itemcf.splitData(dataFile, test_size=0.3)
    # 计算用户之间的相似度
    itemcf.calc_similarity()
    # 评估推荐效果
    # itemcf.evaluate()
    # 查看推荐结果用户
    print("推荐结果", itemcf.recommend(u_index=1))
    print("---", np.where(itemcf.test_mat[1, :] != 0)[0])
