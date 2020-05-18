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
import sys
import math
import random
from operator import itemgetter

# 作用: 使得随机数据可预测
random.seed(0)


class ItemBasedCF():
    ''' TopN recommendation - ItemBasedCF '''

    def __init__(self):
        self.trainset = {}
        self.testset = {}

        # n_sim_user: top 20个用户， n_rec_movie: top 10个推荐结果
        self.n_sim_movie = 20
        self.n_rec_movie = 10

        # user_sim_mat: 电影之间的相似度， movie_popular: 电影的出现次数， movie_count: 总电影数量
        self.movie_sim_mat = {}
        self.movie_popular = {}
        self.movie_count = 0

        print('Similar movie number = %d' % self.n_sim_movie, file=sys.stderr)
        print('Recommended movie number = %d' % self.n_rec_movie, file=sys.stderr)

    @staticmethod
    def loadfile(filename):
        """loadfile(加载文件，返回一个生成器)

        Args:
            filename   文件名
        Returns:
            line       行数据，去空格
        """
        fp = open(filename, 'r')
        for i, line in enumerate(fp):
            yield line.strip('\r\n')
            if i > 0 and i % 100000 == 0:
                print('loading %s(%s)' % (filename, i), file=sys.stderr)
        fp.close()
        print('load %s success' % filename, file=sys.stderr)

    def generate_dataset(self, filename, pivot=0.7):
        """loadfile(加载文件，将数据集按照7:3 进行随机拆分)

        Args:
            filename   文件名
            pivot      拆分比例
        """
        trainset_len = 0
        testset_len = 0

        for line in self.loadfile(filename):
            # 用户ID，电影名称，评分，时间戳
            # user, movie, rating, _ = line.split('::')
            user, movie, rating, _ = line.split('\t')
            # 通过pivot和随机函数比较，然后初始化用户和对应的值
            if (random.random() < pivot):

                # dict.setdefault(key, default=None)
                # key -- 查找的键值
                # default -- 键不存在时，设置的默认键值
                self.trainset.setdefault(user, {})
                self.trainset[user][movie] = int(rating)
                trainset_len += 1
            else:
                self.testset.setdefault(user, {})
                self.testset[user][movie] = int(rating)
                testset_len += 1

        print('分离训练集和测试集成功', file=sys.stderr)
        print('train set = %s' % trainset_len, file=sys.stderr)
        print('test set = %s' % testset_len, file=sys.stderr)

    def calc_movie_sim(self):
        """calc_movie_sim(计算用户之间的相似度)"""

        print('counting movies number and popularity...', file=sys.stderr)

        # 统计在所有的用户中，不同电影的总出现次数， user, movies
        for _, movies in self.trainset.items():
            for movie in movies:
                # count item popularity
                if movie not in self.movie_popular:
                    self.movie_popular[movie] = 0
                self.movie_popular[movie] += 1

        print('count movies number and popularity success', file=sys.stderr)

        # save the total number of movies
        self.movie_count = len(self.movie_popular)
        print('total movie number = %d' % self.movie_count, file=sys.stderr)

        # 统计在相同用户时，不同电影同时出现的次数
        itemsim_mat = self.movie_sim_mat
        print('building co-rated users matrix...', file=sys.stderr)
        # user, movies
        for _, movies in self.trainset.items():
            for m1 in movies:
                for m2 in movies:
                    if m1 == m2:
                        continue
                    itemsim_mat.setdefault(m1, {})
                    itemsim_mat[m1].setdefault(m2, 0)
                    itemsim_mat[m1][m2] += 1
        print('build co-rated users matrix success', file=sys.stderr)

        # calculate similarity matrix
        print('calculating movie similarity matrix...', file=sys.stderr)
        simfactor_count = 0
        PRINT_STEP = 2000000
        for m1, related_movies in itemsim_mat.items():
            for m2, count in related_movies.iteritems():
                # 余弦相似度
                itemsim_mat[m1][m2] = count / math.sqrt(
                    self.movie_popular[m1] * self.movie_popular[m2])
                simfactor_count += 1
                # 打印进度条
                if simfactor_count % PRINT_STEP == 0:
                    print('calculating movie similarity factor(%d)' % simfactor_count, file=sys.stderr)

        print('calculate movie similarity matrix(similarity factor) success', file=sys.stderr)
        print('Total similarity factor number = %d' % simfactor_count, file=sys.stderr)

    # @profile
    def recommend(self, user):
        """recommend(找出top K的电影，对电影进行相似度sum的排序，取出top N的电影数)

        Args:
            user       用户
        Returns:
            rec_movie  电影推荐列表，按照相似度从大到小的排序
        """
        ''' Find K similar movies and recommend N movies. '''
        K = self.n_sim_movie
        N = self.n_rec_movie
        rank = {}
        watched_movies = self.trainset[user]

        # 计算top K 电影的相似度
        # rating=电影评分, w=不同电影出现的次数
        # 耗时分析: 98.2%的时间在 line-154行
        for movie, rating in watched_movies.iteritems():
            for related_movie, w in sorted(
                    self.movie_sim_mat[movie].items(),
                    key=itemgetter(1),
                    reverse=True)[0:K]:
                if related_movie in watched_movies:
                    continue
                rank.setdefault(related_movie, 0)
                rank[related_movie] += w * rating
        # return the N best movies
        return sorted(rank.items(), key=itemgetter(1), reverse=True)[0:N]

    def evaluate(self):
        ''' return precision, recall, coverage and popularity '''
        print('Evaluation start...', file=sys.stderr)

        # 返回top N的推荐结果
        N = self.n_rec_movie
        # varables for precision and recall
        # hit表示命中(测试集和推荐集相同+1)，rec_count 每个用户的推荐数， test_count 每个用户对应的测试数据集的电影数
        hit = 0
        rec_count = 0
        test_count = 0
        # varables for coverage
        all_rec_movies = set()
        # varables for popularity
        popular_sum = 0

        # enumerate将其组成一个索引序列，利用它可以同时获得索引和值
        # 参考地址: http://blog.csdn.net/churximi/article/details/51648388
        for i, user in enumerate(self.trainset):
            if i > 0 and i % 500 == 0:
                print('recommended for %d users' % i, file=sys.stderr)
            test_movies = self.testset.get(user, {})
            rec_movies = self.recommend(user)

            # 对比测试集和推荐集的差异 movie, w
            for movie, _ in rec_movies:
                if movie in test_movies:
                    hit += 1
                all_rec_movies.add(movie)
                # 计算用户对应的电影出现次数log值的sum加和
                popular_sum += math.log(1 + self.movie_popular[movie])
            rec_count += N
            test_count += len(test_movies)

        precision = hit / (1.0 * rec_count)
        recall = hit / (1.0 * test_count)
        coverage = len(all_rec_movies) / (1.0 * self.movie_count)
        popularity = popular_sum / (1.0 * rec_count)

        print('precision=%.4f \t recall=%.4f \t coverage=%.4f \t popularity=%.4f' % (
            precision, recall, coverage, popularity), file=sys.stderr)


if __name__ == '__main__':
    # ratingfile = 'data/16.RecommenderSystems/ml-1m/ratings.dat'
    ratingfile = 'data/16.RecommenderSystems/ml-100k/u.data'

    # 创建ItemCF对象
    itemcf = ItemBasedCF()
    # 将数据按照 7:3的比例，拆分成: 训练集和测试集，存储在usercf的trainset和testset中
    itemcf.generate_dataset(ratingfile, pivot=0.7)
    # 计算用户之间的相似度
    itemcf.calc_movie_sim()
    # 评估推荐效果
    # itemcf.evaluate()
    # 查看推荐结果用户
    user = "2"
    print("推荐结果", itemcf.recommend(user))
    print("---", itemcf.testset.get(user, {}))
