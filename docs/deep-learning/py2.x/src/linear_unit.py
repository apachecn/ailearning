#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# 引入 Perceptron 类
from perceptron import Perceptron

# 定义激活函数 f
f = lambda x: x

class LinearUnit(Perceptron):
    '''
    Desc:
        线性单元类
    Args:
        Perceptron —— 感知器
    Returns:
        None
    '''
    def __init__(self, input_num):
        '''
        Desc:
            初始化线性单元，设置输入参数的个数
        Args:
            input_num —— 输入参数的个数
        Returns:
            None
        '''
        # 初始化我们的感知器类，设置输入参数的个数 input_num 和 激活函数 f
        Perceptron.__init__(self, input_num, f)

# 构造简单的数据集
def get_training_dataset():
    '''
    Desc:
        构建一个简单的训练数据集
    Args:
        None
    Returns:
        input_vecs —— 训练数据集的特征部分
        labels —— 训练数据集的数据对应的标签，是一一对应的
    '''
    # 构建数据集，输入向量列表，每一项是工作年限
    input_vecs = [[5], [3], [8], [1.4], [10.1]]
    # 期望的输出列表，也就是输入向量的对应的标签，与工作年限对应的收入年薪
    labels = [5500, 2300, 7600, 1800, 11400]
    return input_vecs, labels


# 使用我们的训练数据集对线性单元进行训练
def train_linear_unit():
    '''
    Desc:
        使用训练数据集对我们的线性单元进行训练
    Args:
        None
    Returns:
        lu —— 返回训练好的线性单元
    '''
    # 创建感知器对象，输入参数的个数也就是特征数为 1（工作年限）
    lu = LinearUnit(1)
    # 获取构建的数据集
    input_vecs, labels = get_training_dataset()
    # 训练感知器，迭代 10 轮，学习率为 0.01
    lu.train(input_vecs, labels, 10, 0.01)
    # 返回训练好的线性单元
    return lu


# 将图像画出来
def plot(linear_unit):
    '''
    Desc:
        将我们训练好的线性单元对数据的分类情况作图画出来
    Args:
        linear_unit —— 训练好的线性单元
    Returns:
        None
    '''
    # 引入绘图的库
    import matplotlib.pyplot as plt
    # 获取训练数据：特征 input_vecs 与 对应的标签 labels
    input_vecs, labels = get_training_dataset()
    # figure() 创建一个 Figure 对象，与用户交互的整个窗口，这个 figure 中容纳着 subplots
    fig = plt.figure()
    # 在 figure 对象中创建 1行1列中的第一个图
    ax = fig.add_subplot(111)
    # scatter(x, y) 绘制散点图，其中的 x,y 是相同长度的数组序列
    ax.scatter(map(lambda x: x[0], input_vecs), labels)
    # 设置权重
    weights = linear_unit.weights
    # 设置偏置项
    bias = linear_unit.bias
    # range(start, stop, step) 从 start 开始，到 stop 结束，步长为 step
    x = range(0, 12, 1)
    # 计算感知器对输入计算得到的值
    y = map(lambda x: weights[0] * x + bias, x)
    # 将图画出来
    ax.plot(x, y)
    # 将最终的图展示出来
    plt.show()


if __name__ == '__main__':
    '''
    Desc:
        main 函数，训练我们的线性单元，并进行预测
    Args:
        None
    Returns:
        None
    '''
    # 首先训练我们的线性单元
    linear_unit = train_linear_unit()
    # 打印训练获得的权重 和 偏置
    print linear_unit
    # 测试
    print 'Work 3.4 years, monthly salary = %.2f' % linear_unit.predict([3.4])
    print 'Work 15 years, monthly salary = %.2f' % linear_unit.predict([15])
    print 'Work 1.5 years, monthly salary = %.2f' % linear_unit.predict([1.5])
    print 'Work 6.3 years, monthly salary = %.2f' % linear_unit.predict([6.3])
    plot(linear_unit)
