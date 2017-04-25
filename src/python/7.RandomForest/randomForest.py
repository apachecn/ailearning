#!/usr/bin/python
# coding:utf8

'''
Created 2017-04-25
Random Forest Algorithm on Sonar Dataset
@author: Flying_sfeng/jiangzhonglian
'''
from random import seed, randrange, random
from math import sqrt
from math import log


# 导入csv文件
def loadDataSet(filename):
    dataset = []
    with open(filename, 'r') as fr:
        for line in fr.readlines():
            if not line:
                continue
            lineArr = []
            for featrue in line.split(','):
                # strip()返回移除字符串头尾指定的字符生成的新字符串
                str_f = featrue.strip()
                if str_f.isdigit():
                    # 将数据集的第column列转换成float形式
                    lineArr.append(float(str_f))
                else:
                    # 添加分类标签
                    lineArr.append(str_f)
            dataset.append(lineArr)
    return dataset


# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):  #将数据集dataset分成n_flods份，每份包含len(dataset) / n_folds个值，每个值由dataset数据集的内容随机产生，每个值被使用一次
    dataset_split = list()
    dataset_copy = list(dataset)  #复制一份dataset,防止dataset的内容改变
    fold_size = len(dataset) / n_folds
    for i in range(n_folds):
        fold = list()   #每次循环fold清零，防止重复导入dataset_split
        while len(fold) < fold_size:   #这里不能用if，if只是在第一次判断时起作用，while执行循环，直到条件不成立
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))  #将对应索引index的内容从dataset_copy中导出，并将该内容从dataset_copy中删除。pop() 函数用于移除列表中的一个元素（默认最后一个元素），并且返回该元素的值。
        dataset_split.append(fold)
    return dataset_split    #由dataset分割出的n_folds个数据构成的列表，为了用于交叉验证


# Calculate accuracy percentage  
def accuracy_metric(actual, predicted):  #导入实际值和预测值，计算精确度
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# Split a dataset based on an attribute and an attribute value #根据特征和特征值分割数据集
def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


# Calculate the Gini index for a split dataset
def gini_index(groups, class_values):   #个人理解：计算代价，分类越准确，则gini越小
    gini = 0.0
    for class_value in class_values:  #class_values =[0,1] 
        for group in groups:          #groups=(left,right)
            size = len(group)
            if size == 0:
                continue
            proportion = [row[-1] for row in group].count(class_value) / float(size)
            gini += (proportion * (1.0 - proportion))  #个人理解：计算代价，分类越准确，则gini越小
    return gini


# Select the best split point for a dataset  #找出分割数据集的最优特征，得到最优的特征index，特征值row[index]，以及分割完的数据groups（left,right）
def get_split(dataset, n_features):
    class_values = list(set(row[-1] for row in dataset))  #class_values =[0,1]
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    features = list()
    while len(features) < n_features:   
        index = randrange(len(dataset[0])-1)  #往features添加n_features个特征（n_feature等于特征数的根号），特征索引从dataset中随机取
        if index not in features:
            features.append(index)
    for index in features:        #在n_features个特征中选出最优的特征索引，并没有遍历所有特征，从而保证了每课决策树的差异性
        for row in dataset:
            groups = test_split(index, row[index], dataset)  #groups=(left,right)；row[index]遍历每一行index索引下的特征值作为分类值value，找出最优的分类特征和特征值
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups  #最后得到最优的分类特征b_index,分类特征值b_value,分类结果b_groups。b_value为分错的代价成本。
    #print b_score
    return {'index':b_index, 'value':b_value, 'groups':b_groups}


# Create a terminal node value #输出group中出现次数较多的标签
def to_terminal(group):
    outcomes = [row[-1] for row in group]           #max()函数中，当key参数不为空时，就以key的函数对象为判断的标准;
    return max(set(outcomes), key=outcomes.count)   # 输出group中出现次数较多的标签  


# Create child splits for a node or make terminal  #创建子分割器，递归分类，直到分类结束
def split(node, max_depth, min_size, n_features, depth):  #max_depth = 10，min_size = 1，n_features = int(sqrt(len(dataset[0])-1)) 
    left, right = node['groups']
    del(node['groups'])
# check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
# check for max depth
    if depth >= max_depth:   #max_depth=10表示递归十次，若分类还未结束，则选取数据中分类标签较多的作为结果，使分类提前结束，防止过拟合
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
# process left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left, n_features)  #node['left']是一个字典，形式为{'index':b_index, 'value':b_value, 'groups':b_groups}，所以node是一个多层字典
        split(node['left'], max_depth, min_size, n_features, depth+1)  #递归，depth+1计算递归层数
# process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right, n_features)
        split(node['right'], max_depth, min_size, n_features, depth+1)


# Build a decision tree
def build_tree(train, max_depth, min_size, n_features):
    #root = get_split(dataset, n_features)
    root = get_split(train, n_features)
    split(root, max_depth, min_size, n_features, 1)
    return root


# Make a prediction with a decision tree
def predict(node, row):   #预测模型分类结果
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):    #isinstance是Python中的一个内建函数。是用来判断一个对象是否是一个已知的类型。
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


# Make a prediction with a list of bagged trees
def bagging_predict(trees, row):
    predictions = [predict(tree, row) for tree in trees]  #使用多个决策树trees对测试集test的第row行进行预测，再使用简单投票法判断出该行所属分类
    return max(set(predictions), key=predictions.count)


# Create a random subsample from the dataset with replacement
def subsample(dataset, ratio):   #创建数据集的随机子样本
    sample = list()
    n_sample = round(len(dataset) * ratio)   #round() 方法返回浮点数x的四舍五入值。
    while len(sample) < n_sample:
        index = randrange(len(dataset))  #有放回的随机采样，有一些样本被重复采样，从而在训练集中多次出现，有的则从未在训练集中出现，此则自助采样法。从而保证每棵决策树训练集的差异性
        sample.append(dataset[index])
    return sample


# Random Forest Algorithm
def random_forest(train, test, max_depth, min_size, sample_size, n_trees, n_features):
    trees = list()
    for i in range(n_trees):   #n_trees表示决策树的数量
        sample = subsample(train, sample_size)  #随机采样保证了每棵决策树训练集的差异性
        tree = build_tree(sample, max_depth, min_size, n_features)  #建立一个决策树
        trees.append(tree)
    predictions = [bagging_predict(trees, row) for row in test]
    return(predictions)


# Evaluate an algorithm using a cross validation split   
def evaluate_algorithm(dataset, algorithm, n_folds, *args):   #评估算法性能，返回模型得分
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:   #每次循环从folds从取出一个fold作为测试集，其余作为训练集，遍历整个folds，实现交叉验证
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])   #将多个fold列表组合成一个train_set列表
        test_set = list()
        for row in fold:   #fold表示从原始数据集dataset提取出来的测试集
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


if __name__ == '__main__':

    # 加载数据
    dataset = loadDataSet('input/7.RandomForest/sonar-all-data.txt')
    # print dataset

    n_folds = 5   #分成5份数据，进行交叉验证
    #max_depth = 10 #递归十次
    max_depth = 20 #调参（自己修改） #决策树深度不能太深，不然容易导致过拟合
    min_size = 1
    sample_size = 1.0
    #n_features = int(sqrt(len(dataset[0])-1))
    n_features =15  #调参（自己修改） #准确性与多样性之间的权衡
    for n_trees in [1,5,10]:  #理论上树是越多越好
        scores = evaluate_algorithm(dataset, random_forest, n_folds, max_depth, min_size, sample_size, n_trees, n_features)
        # 每一次执行本文件时都能产生同一个随机数
        seed(1)
        print 'random=', random()
        print('Trees: %d' % n_trees)
        print('Scores: %s' % scores)
        print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
