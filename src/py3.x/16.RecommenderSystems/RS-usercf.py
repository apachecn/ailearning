#!/usr/bin/python
# coding:utf8
'''
Created on 2015-06-22
Update  on 2017-05-16
Author: Lockvictor/片刻
《推荐系统实践》协同过滤算法源代码
参考地址：https://github.com/Lockvictor/MovieLens-RecSys
更新地址：https://github.com/apachecn/AiLearning
'''
import sys
import math
import random
from operator import itemgetter
print(__doc__)
# 作用：使得随机数据可预测
random.seed(0)


class UserBasedCF():
    ''' TopN recommendation - UserBasedCF '''

    def __init__(self):
        self.trainset = {}
        self.testset = {}

        # n_sim_user: top 20个用户， n_rec_movie: top 10个推荐结果
        self.n_sim_user = 20
        self.n_rec_movie = 10

        # user_sim_mat: 用户之间的相似度， movie_popular: 电影的出现次数， movie_count: 总电影数量
        self.user_sim_mat = {}
        self.movie_popular = {}
        self.movie_count = 0

        print >> sys.stderr, 'similar user number = %d' % self.n_sim_user
        print >> sys.stderr, 'recommended movie number = %d' % self.n_rec_movie

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
                print >> sys.stderr, 'loading %s(%s)' % (filename, i)
        fp.close()
        print >> sys.stderr, 'load %s success' % filename

    def generate_dataset(self, filename, pivot=0.7):
        """loadfile(加载文件，将数据集按照7:3 进行随机拆分)

        Args:
            filename   文件名
            pivot      拆分比例
        """
        trainset_len = 0
        testset_len = 0

        for line in self.loadfile(filename):
            # 用户ID，电影名称，评分，时间戳timestamp
            # user, movie, rating, timestamp = line.split('::')
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

        print >> sys.stderr, '分离训练集和测试集成功'
        print >> sys.stderr, 'train set = %s' % trainset_len
        print >> sys.stderr, 'test  set = %s' % testset_len

    def calc_user_sim(self):
        """calc_user_sim(计算用户之间的相似度)"""

        # build inverse table for item-users
        # key=movieID, value=list of userIDs who have seen this movie
        print >> sys.stderr, 'building movie-users inverse table...'
        movie2users = dict()

        # 同一个电影中，收集用户的集合
        # 统计在所有的用户中，不同电影的总出现次数
        for user, movies in self.trainset.items():
            for movie in movies:
                # inverse table for item-users
                if movie not in movie2users:
                    movie2users[movie] = set()
                movie2users[movie].add(user)
                # count item popularity at the same time
                if movie not in self.movie_popular:
                    self.movie_popular[movie] = 0
                self.movie_popular[movie] += 1

        print >> sys.stderr, 'build movie-users inverse table success'

        # save the total movie number, which will be used in evaluation
        self.movie_count = len(movie2users)
        print >> sys.stderr, 'total movie number = %d' % self.movie_count

        usersim_mat = self.user_sim_mat
        # 统计在相同电影时，不同用户同时出现的次数
        print >> sys.stderr, 'building user co-rated movies matrix...'

        for movie, users in movie2users.items():
            for u in users:
                for v in users:
                    if u == v:
                        continue
                    usersim_mat.setdefault(u, {})
                    usersim_mat[u].setdefault(v, 0)
                    usersim_mat[u][v] += 1
        print >> sys.stderr, 'build user co-rated movies matrix success'

        # calculate similarity matrix
        print >> sys.stderr, 'calculating user similarity matrix...'
        simfactor_count = 0
        PRINT_STEP = 2000000
        for u, related_users in usersim_mat.items():
            for v, count in related_users.iteritems():
                # 余弦相似度
                usersim_mat[u][v] = count / math.sqrt(
                    len(self.trainset[u]) * len(self.trainset[v]))
                simfactor_count += 1
                # 打印进度条
                if simfactor_count % PRINT_STEP == 0:
                    print >> sys.stderr, 'calculating user similarity factor(%d)' % simfactor_count

        print >> sys.stderr, 'calculate user similarity matrix(similarity factor) success'
        print >> sys.stderr, 'Total similarity factor number = %d' % simfactor_count

    # @profile
    def recommend(self, user):
        """recommend(找出top K的用户，所看过的电影，对电影进行相似度sum的排序，取出top N的电影数)

        Args:
            user       用户
        Returns:
            rec_movie  电影推荐列表，按照相似度从大到小的排序
        """
        ''' Find K similar users and recommend N movies. '''
        K = self.n_sim_user
        N = self.n_rec_movie
        rank = dict()
        watched_movies = self.trainset[user]

        # 计算top K 用户的相似度
        # v=similar user, wuv=不同用户同时出现的次数，根据wuv倒序从大到小选出K个用户进行排列
        # 耗时分析：50.4%的时间在 line-160行
        for v, wuv in sorted(
                self.user_sim_mat[user].items(), key=itemgetter(1),
                reverse=True)[0:K]:
            for movie, rating in self.trainset[v].iteritems():
                if movie in watched_movies:
                    continue
                # predict the user's "interest" for each movie
                rank.setdefault(movie, 0)
                rank[movie] += wuv * rating
        # return the N best movies
        """
        wuv
        precision=0.3766         recall=0.0759   coverage=0.3183         popularity=6.9194

        wuv * rating
        precision=0.3865         recall=0.0779   coverage=0.2681         popularity=7.0116
        """
        return sorted(rank.items(), key=itemgetter(1), reverse=True)[0:N]

    def evaluate(self):
        ''' return precision, recall, coverage and popularity '''
        print >> sys.stderr, 'Evaluation start...'

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
        # 参考地址：http://blog.csdn.net/churximi/article/details/51648388
        for i, user in enumerate(self.trainset):
            if i > 0 and i % 500 == 0:
                print >> sys.stderr, 'recommended for %d users' % i
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

        print >> sys.stderr, 'precision=%.4f \t recall=%.4f \t coverage=%.4f \t popularity=%.4f' % (
            precision, recall, coverage, popularity)


if __name__ == '__main__':
    # ratingfile = 'db/16.RecommenderSystems/ml-1m/ratings.dat'
    ratingfile = 'db/16.RecommenderSystems/ml-100k/u.data'

    # 创建UserCF对象
    usercf = UserBasedCF()
    # 将数据按照 7:3的比例，拆分成：训练集和测试集，存储在usercf的trainset和testset中
    usercf.generate_dataset(ratingfile, pivot=0.7)
    # 计算用户之间的相似度
    usercf.calc_user_sim()
    # 评估推荐效果
    usercf.evaluate()
