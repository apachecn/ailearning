#!/usr/bin/python
# coding:utf-8
# -------------------------------------------------------------------------------
# Name:    推荐系统
# Purpose: 基于内容推荐
# Author:  jiangzhonglian
# Create_time:  2020年10月15日
# Update_time:  2020年10月21日
# -------------------------------------------------------------------------------
import os
import sys
import numpy as np
import pandas as pd
# 自定义库
import config.set_content as setting
from middleware.utils import pd_load, pd_like, pd_save, pd_rename, get_days


def data_converting(infile, outfile):
    """
    # 将用户交易数据转化为: 
    # 将
    #     用户ID、各种基金、变动金额、时间
    # 转化为：
    #     用户ID、基金ID、购买金额、时间的数据
    """
    print("Loading user daliy data...")
    df = pd_load(infile)
    df["money"] = df["变动金额"].apply(lambda line: abs(line))
    df_user_item = df.groupby(["用户账号", "证券代码"], as_index=False).agg({
            "money": np.sum
        }).sort_values("money", ascending=True)
    pd_rename(df_user_item, ["user_id", "item_id", "rating"])
    pd_save(df_user_item, outfile)


def create_user2item(infile, outfile):
    """创建user-item评分矩阵"""

    print("Loading user daliy data...")
    df_user_item = pd_load(infile)

    user_id = sorted(df_user_item['user_id'].unique(), reverse=False)
    item_id = sorted(df_user_item['item_id'].unique(), reverse=False)
    # print("+++ user_id:", user_id)
    # print("+++ item_id:", item_id)
    rating_matrix = np.zeros([len(user_id),len(item_id)])
    rating_matrix = pd.DataFrame(rating_matrix, index=user_id, columns=item_id)

    print("Converting data...")
    count = 0
    user_num= len(user_id)
    for uid in user_id:
        user_rating = df_user_item[df_user_item['user_id'] == uid].drop(['user_id'], axis=1)
        user_rated_num = len(user_rating)
        for row in range(0, user_rated_num):
            item_id = user_rating['item_id'].iloc[row]
            # 行（用户），列（电影），得分
            rating_matrix.loc[uid, item_id] = user_rating['rating'].iloc[row]

        count += 1
        if count % 10 == 0:
            completed_percentage = round(float(count) / user_num * 100)
            print("Completed %s" % completed_percentage + "%")

    rating_matrix.index.name = 'user_id'
    pd_save(rating_matrix, outfile, index=True)


def create_item2feature(infile, outfile):
    """创建 item-特征-是否存在 矩阵"""

    print("Loading item feature data...")
    df_item_info = pd_load(infile, header=1)
    items_num = df_item_info.shape[0]
    columns = df_item_info.columns.tolist()
    new_cols = [col for col in columns if col not in ["info_type", "info_investype"]]
    info_types      = sorted(df_item_info["info_type"].unique(), reverse=False)
    info_investypes = sorted(df_item_info["info_investype"].unique(), reverse=False)
    dict_n_cols = {"info_type": info_types, "info_investype": info_investypes}
    new_cols.append(dict_n_cols)
    # 获取新的列名
    def get_new_columns(new_cols):
        new_columns = []
        for col in new_cols:
            if isinstance(col, dict):
                for k, vs in col.items():
                    new_columns += vs
            else:
                new_columns.append(col)
        return new_columns
    new_columns = get_new_columns(new_cols)
    # print(new_cols)
    # print(new_columns)

    # ['item_id', 'info_name', 'info_trackerror', 'info_manafeeratioo', 'info_custfeeratioo', 'info_salefeeratioo', 'info_foundsize', 'info_foundlevel', 'info_creattime', 'info_unitworth'
    # {'info_type': ['QDII-ETF', '混合型', '股票指数', 'ETF-场内'], 'info_investype': ['契约型开放式', '契约型封闭式']}]
    def deal_line(line, new_cols):
        result = []
        for col in new_cols:
            if isinstance(col, str):
                result.append(line[col])
            elif isinstance(col, dict):
                for k, vs in col.items():
                    for v in vs:
                        if v == line[k]:
                            result.append(1)
                        else:
                            result.append(0)
            else:
                print("类型错误")
                sys.exit(1)
        return result

    df = df_item_info.apply(lambda line: deal_line(line, new_cols), axis=1, result_type="expand")
    pd_rename(df, new_columns)
    # 处理时间
    end_time = "2020-10-19"
    df["days"] = df["info_creattime"].apply(lambda str_time: get_days(str_time, end_time))
    # print(df.head(5))
    df.drop(['info_name', 'info_foundlevel', 'info_creattime'], axis=1, inplace=True)
    pd_save(df, outfile)


def rs_1_data_preprocess():
    # 原属数据
    data_infile = setting.PATH_CONFIG["user_daily"]
    # 用户-物品-评分
    user_infile = setting.PATH_CONFIG["user_item"]
    user_outfile = setting.PATH_CONFIG["matrix_user_item2rating"]
    # 物品-特征-评分
    item_infile = setting.PATH_CONFIG["item_info"]
    item_outfile = setting.PATH_CONFIG["matrix_item2feature"]

    # 判断用户交易数据，如果不存在就要重新生成
    if not os.path.exists(user_infile):
        """数据处理部分"""
        # user 数据预处理
        data_converting(data_infile, user_infile)
        # 创建 user-item-评分 矩阵
        create_user2item(user_infile, user_outfile)
    else:
        if not os.path.exists(user_outfile):
            # 创建 user-item-评分 矩阵
            create_user2item(user_infile, user_outfile)

    if not os.path.exists(item_outfile):
        # 创建 item-feature-是否存在 矩阵
        create_item2feature(item_infile, item_outfile)

    user_feature = pd_load(user_outfile)
    item_feature = pd_load(item_outfile)
    user_feature.set_index('user_id', inplace=True)
    item_feature.set_index('item_id', inplace=True)
    return user_feature, item_feature


def cos_measure(item_feature_vector, user_rated_items_matrix):
    """
    计算item之间的余弦夹角相似度
    :param item_feature_vector: 待测量的item特征向量
    :param user_rated_items_matrix: 用户已评分的items的特征矩阵
    :return: 待计算item与用户已评分的items的余弦夹角相识度的向量
    """
    x_c = (item_feature_vector * user_rated_items_matrix.T) + 0.0000001
    mod_x = np.sqrt(item_feature_vector * item_feature_vector.T)
    mod_c = np.sqrt((user_rated_items_matrix * user_rated_items_matrix.T).diagonal())
    cos_xc = x_c / (mod_x * mod_c)

    return cos_xc


def comp_user_feature(user_rated_vector, item_feature_matrix):
    """
    根据user的评分来计算得到user的喜好特征
    :param user_rated_vector  : user的评分向量
    :param item_feature_matrix: item的特征矩阵
    :return: user的喜好特征
    """
    # user评分的均值
    user_rating_mean = user_rated_vector.mean()
    # # 分别得到user喜欢和不喜欢item的向量以及item对应的引索(以该user的评分均值来划分)
    user_like_item = user_rated_vector.loc[user_rated_vector >= user_rating_mean]
    user_unlike_item = user_rated_vector.loc[user_rated_vector < user_rating_mean]

    print("user_like_item: \n", user_like_item)
    print("user_unlike_item: \n", user_unlike_item)

    # 获取买入和卖出的 index
    user_like_item_index = map(int, user_like_item.index.values)
    user_unlike_item_index = map(int, user_unlike_item.index.values)
    # 获取买入和卖出的 value
    user_like_item_rating = np.matrix(user_like_item.values)
    user_unlike_item_rating = np.matrix(user_unlike_item.values)

    #得到user喜欢和不喜欢item的特征矩阵
    user_like_item_feature_matrix = np.matrix(item_feature_matrix.loc[user_like_item_index, :].values)
    user_unlike_item_feature_matrix = np.matrix(item_feature_matrix.loc[user_unlike_item_index, :].values)

    #计算user的喜好特征向量，以其对item的评分作为权重
    weight_of_like = user_like_item_rating / user_like_item_rating.sum()
    weight_of_unlike = user_unlike_item_rating / user_unlike_item_rating.sum()

    print("weight_of_like: ", weight_of_like)
    print("weight_of_unlike: ", weight_of_unlike)

    #计算user的喜欢特征和不喜欢特征以及总特征
    user_like_feature = weight_of_like * user_like_item_feature_matrix
    user_unlike_feature = weight_of_unlike * user_unlike_item_feature_matrix
    user_feature_tol = user_like_feature - user_unlike_feature
    return user_feature_tol


def rs_2_cb_recommend(user_feature, item_feature_matrix, K=20):
    """
    计算得到与user最相似的Top K个item推荐给user
    :param user_feature: 待推荐用户的对item的评分向量
    :param item_feature_matrix: 包含所有item的特征矩阵
    :param K: 推荐给user的item数量
    :return: 与user最相似的Top K个item的编号
    """
    # 得到user已评分和未评分的item向量
    user_rated_vector = user_feature.loc[user_feature > 0]
    # print("操作 >>> \n", user_rated_vector)
    # user_unrated_vector = user_feature.loc[user_feature == 0]
    # print("未操作 >>> \n", user_unrated_vector)
    # 买过的其实也可以推荐
    user_unrated_vector = user_feature
    # print(">>> \n", user_unrated_vector)

    # user喜好总特征(就是用户的调性)
    user_item_feature_tol = comp_user_feature(user_rated_vector, item_feature_matrix)
    print(">>> 用户调性", user_item_feature_tol)
    #未评分item的特征矩阵
    user_unrated_item_index = map(int, user_unrated_vector.index.values)
    user_unrated_item_feature_matrix = np.matrix(item_feature_matrix.loc[user_unrated_item_index, :].values)

    #得到相似度并进行排序
    similarity = list(np.array(cos_measure(user_item_feature_tol, user_unrated_item_feature_matrix))[0])

    key = {'item_index': list(user_unrated_vector.index.values), 'similarity': similarity}
    item_sim_df = pd.DataFrame(key)
    item_sim_df.sort_values('similarity', ascending=False, inplace=True)
    # print(item_sim_df.head(100))
    return item_sim_df.iloc[:K, 0].values


def estimate_rate(user_rated_vector, similarity):
    """
    估计用户对item的评分
    :param user_rated_vector: 用户已有item评分向量
    :param similarity: 待估计item和已评分item的相识度向量
    :return:用户对item的评分的估计
    """
    rate_hat = (user_rated_vector * similarity.T) / similarity.sum()
    # print(">>> ", rate_hat)
    return rate_hat[0, 0]


def rs_2_cb_recommend_estimate(user_feature, item_feature_matrix, item):
    """
    基于内容的推荐算法对item的评分进行估计
    :param item_feature_matrix: 包含所有item的特征矩阵
    :param user_feature: 待估计用户的对item的评分向量
    :param item: 待估计item的编号
    :return: 基于内容的推荐算法对item的评分进行估计
    """
    # #得到item的引索以及特征矩阵
    # item_index = item_feature_matrix.index
    # item_feature = item_feature_matrix.values

    #得到所有user评分的item的引索
    user_item_index = user_feature.index

    #某一用户已有评分item的评分向量和引索以及item的评分矩阵
    user_rated_vector = np.matrix(user_feature.loc[user_feature > 0].values)
    user_rated_items = map(int, user_item_index[user_feature > 0].values)

    user_rated_items_matrix = np.matrix(item_feature_matrix.loc[user_rated_items, :].values)

    #待评分item的特征向量，函数中给出的是该item的Id
    item_feature_vector = np.matrix(item_feature_matrix.loc[item].values)

    #得到待计算item与用户已评分的items的余弦夹角相识度的向量
    cos_xc = cos_measure(item_feature_vector, user_rated_items_matrix)
    # print(">>> 相似度: %s" % cos_xc)
    #计算uesr对该item的评分估计
    rate_hat = estimate_rate(user_rated_vector, cos_xc)
    return rate_hat


def main():
    # 数据初始化
    user_id = 20200930
    K = 10
    user_feature, item_feature = rs_1_data_preprocess()
    # 基于内容推荐的模块(给某一个用户推荐 10个 他感兴趣的内容)
    user_feature = user_feature.loc[user_id, :]  # 一行 用户(具体某一个用户)-电影-评分 数据
    print(">>> 1 \n", user_feature)
    # 效果不好，有几个原因
    # 1. 交易数据比较少
    # 2. 基金的特征不够全面
    # 3. 优化喜欢和不喜欢的阈值
    result = rs_2_cb_recommend(user_feature, item_feature, K)
    print(result)
    # for code in result:
    #     # 给某一个用户推荐一个item, 预估推荐评分
    #     price = rs_2_cb_recommend_estimate(user_feature, item_feature, code)
    #     if price > 1000:
    #         print("--- %s 基金买入 %s" % (code, abs(price)) )
    #     elif price < -1000:
    #         print("--- %s 基金卖出 %s" % (code, abs(price)) )
    #     else:
    #         print("--- 不做任何操作")


if __name__ == "__main__":
    main()
