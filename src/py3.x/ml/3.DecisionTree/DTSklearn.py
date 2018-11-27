#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 原始链接： http://blog.csdn.net/lsldd/article/details/41223147
# GitHub: https://github.com/apachecn/AiLearning
import numpy as np
from sklearn import tree
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def createDataSet():
    ''' 数据读入 '''
    data = []
    labels = []
    with open("db/3.DecisionTree/data.txt") as ifile:
        for line in ifile:
            # 特征： 身高 体重   label： 胖瘦
            tokens = line.strip().split(' ')
            data.append([float(tk) for tk in tokens[:-1]])
            labels.append(tokens[-1])
    # 特征数据
    x = np.array(data)
    # label分类的标签数据
    labels = np.array(labels)
    # 预估结果的标签数据
    y = np.zeros(labels.shape)

    ''' 标签转换为0/1 '''
    y[labels == 'fat'] = 1
    print(data, '-------', x, '-------', labels, '-------', y)
    return x, y


def predict_train(x_train, y_train):
    '''
    使用信息熵作为划分标准，对决策树进行训练
    参考链接： http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
    '''
    clf = tree.DecisionTreeClassifier(criterion='entropy')
    # print(clf)
    clf.fit(x_train, y_train)
    ''' 系数反映每个特征的影响力。越大表示该特征在分类中起到的作用越大 '''
    print('feature_importances_: %s' % clf.feature_importances_)

    '''测试结果的打印'''
    y_pre = clf.predict(x_train)
    # print(x_train)
    print(y_pre)
    print(y_train)
    print(np.mean(y_pre == y_train))
    return y_pre, clf


def show_precision_recall(x, y, clf,  y_train, y_pre):
    '''
    准确率与召回率
    参考链接： http://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html#sklearn.metrics.precision_recall_curve
    '''
    precision, recall, thresholds = precision_recall_curve(y_train, y_pre)
    # 计算全量的预估结果
    answer = clf.predict_proba(x)[:, 1]

    '''
    展现 准确率与召回率
        precision 准确率
        recall 召回率
        f1-score  准确率和召回率的一个综合得分
        support 参与比较的数量
    参考链接：http://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html#sklearn.metrics.classification_report
    '''
    # target_names 以 y的label分类为准
    target_names = ['thin', 'fat']
    print(classification_report(y, answer, target_names=target_names))
    print(answer)
    print(y)


def show_pdf(clf):
    '''
    可视化输出
    把决策树结构写入文件: http://sklearn.lzjqsdd.com/modules/tree.html

    Mac报错：pydotplus.graphviz.InvocationException: GraphViz's executables not found
    解决方案：sudo brew install graphviz
    参考写入： http://www.jianshu.com/p/59b510bafb4d
    '''
    # with open("testResult/tree.dot", 'w') as f:
    #     from sklearn.externals.six import StringIO
    #     tree.export_graphviz(clf, out_file=f)

    import pydotplus
    from sklearn.externals.six import StringIO
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("../../../output/3.DecisionTree/tree.pdf")

    # from IPython.display import Image
    # Image(graph.create_png())


if __name__ == '__main__':
    x, y = createDataSet()

    ''' 拆分训练数据与测试数据， 80%做训练 20%做测试 '''
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    print('拆分数据：', x_train, x_test, y_train, y_test)

    # 得到训练的预测结果集
    y_pre, clf = predict_train(x_train, y_train)

    # 展现 准确率与召回率
    show_precision_recall(x, y, clf, y_train, y_pre)

    # 可视化输出
    show_pdf(clf)
