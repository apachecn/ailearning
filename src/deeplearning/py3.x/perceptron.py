#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# 神经元 / 感知器

class Perceptron():
    '''
    Desc:
        感知器类
    Args:
        None
    Returns:
        None
    '''

    def __init__(self, input_num, activator):
        '''
        Desc:
            初始化感知器
        Args:
            input_num —— 输入参数的个数
            activator —— 激活函数
        Returns:
            None
        '''
        # 设置的激活函数
        self.activator = activator
        # 权重向量初始化为 0
        self.weights = [0.0 for _ in range(input_num)]
        # 偏置项初始化为 0
        self.bias = 0.0

    
    def __str__(self):
        '''
        Desc:
            将感知器信息打印出来
        Args:
            None
        Returns:
            None
        '''
        return('weights\t:%s\nbias\t:%f\n' % (self.weights, self.bias))


    def predict(self, input_vec):
        '''
        Desc:
            输入向量，输出感知器的计算结果
        Args:
            input_vec —— 输入向量
        Returns:
            感知器的计算结果
        '''
        # 将输入向量的计算结果返回
        # 调用 激活函数 activator ，将输入向量输入，计算感知器的结果
        # reduce() 函数是 python 2 的内置函数，从 python 3 开始移到了 functools 模块
        # reduce() 从左到右对一个序列的项累计地应用有两个参数的函数，以此合并序列到一个单一值，例如 reduce(lambda x,y: x+y, [1,2,3,4,5]) 计算的就是 ((((1+2)+3)+4)+5)
        # map() 接收一个函数 f 和一个 list ，并通过把函数 f 依次作用在 list 的每个元素上，得到一个新的 list 返回。比如我们的 f 函数是计算平方， map(f, [1,2,3,4,5]) ===> 返回 [1,4,9,16,25]
        # zip() 接收任意多个（包括 0 个和 1个）序列作为参数，返回一个 tuple 列表。例：x = [1,2,3] y = [4,5,6] z = [7,8,9] xyz = zip(x, y, z) ===> [(1,4,7), (2,5,8), (3,6,9)]
        return self.activator(reduce(lambda a, b: a + b,map(lambda (x, w): x * w, zip(input_vec, self.weights)), 0.0) + self.bias)


    def train(self, input_vecs, labels, iteration, rate):
        '''
        Desc:
            输入训练数据：一组向量、与每个向量对应的 label; 以及训练轮数、学习率
        Args:
            input_vec —— 输入向量
            labels —— 数据对应的标签
            iteration —— 训练的迭代轮数
            rate —— 学习率
        Returns:
            None
        '''
        for i in range(iteration):
            self._one_iteration(input_vecs, labels, rate)

    
    def _one_iteration(self, input_vecs, labels, rate):
        '''
        Desc:
            训练过程的一次迭代过程
        Args:
            input_vecs —— 输入向量
            labels —— 数据对应的标签
            rate —— 学习率
        Returns:
            None
        '''
        # zip() 接收任意多个（包括 0 个和 1个）序列作为参数，返回一个 tuple 列表。例：x = [1,2,3] y = [4,5,6] z = [7,8,9] xyz = zip(x, y, z) ===> [(1,4,7), (2,5,8), (3,6,9)]
        samples = zip(input_vecs, labels)
        # 对每个样本，按照感知器规则更新权重
        for (input_vec, label) in samples:
            # 计算感知器在当前权重下的输出
            output = self.predict(input_vec)
            # 更新权重
            output = self._update_weights(input_vec, output, label, rate)

    def _update_weights(self, input_vec, output, label, rate):
        '''
        Desc:
            按照感知器规则更新权重
        Args:
            input_vec —— 输入向量
            output —— 经过感知器规则计算得到的输出
            label —— 输入向量对应的标签
            rate —— 学习率
        Returns:
            None
        '''
        # 利用感知器规则更新权重
        delta = label - output
        # map() 接收一个函数 f 和一个 list ，并通过把函数 f 依次作用在 list 的每个元素上，得到一个新的 list 返回。比如我们的 f 函数是计算平方， map(f, [1,2,3,4,5]) ===> 返回 [1,4,9,16,25]
        # zip() 接收任意多个（包括 0 个和 1个）序列作为参数，返回一个 tuple 列表。例：x = [1,2,3] y = [4,5,6] z = [7,8,9] xyz = zip(x, y, z) ===> [(1,4,7), (2,5,8), (3,6,9)]
        self.weights = map(lambda (x, w): w + rate * delta * x, zip(input_vec, self.weights))
        # 更新 bias
        self.bias += rate * delta

    

def f(x):
    '''
    Desc:
        定义激活函数 f
    Args:
        x —— 输入向量
    Returns:
        （实现阶跃函数）大于 0 返回 1，否则返回 0
    '''
    return 1 if x > 0 else 0


def get_training_dataset():
    '''
    Desc:
        基于 and 真值表来构建/获取训练数据集
    Args:
        None
    Returns:
        input_vecs —— 输入向量
        labels —— 输入向量对应的标签
    '''
    # 构建训练数据，输入向量的列表
    input_vecs = [[1,1],[0,0],[1,0],[0,1]]
    # 期望的输出列表，也就是上面的输入向量的列表中数据对应的标签，是一一对应的
    labels = [1, 0, 0, 0]
    return input_vecs, labels


def train_and_perceptron():
    '''
    Desc:
        使用 and 真值表来训练我们的感知器
    Args:
        None
    Returns:
        p —— 返回训练好的感知器
    '''
    # 创建感知器，输入参数的个数是 2 个（因为 and 是个二元函数），激活函数为 f
    p = Perceptron(2, f)
    # 进行训练，迭代 10 轮，学习速率是我们设定的 rate ，为 0.1
    input_vecs, labels = get_training_dataset()
    p.train(input_vecs, labels, 10, 0.1)
    # 返回训练好的感知器
    return p


if __name__ == '__main__':
    '''
    Desc:
        主函数，调用上面返回的训练好的感知器进行预测
    Args:
        None
    Returns:
        None
    '''
    # 训练 and 感知器
    and_perceptron = train_and_perceptron()
    # 打印训练获得的权重
    print and_perceptron
    # 测试
    print '1 and 1 = %d' % and_perceptron.predict([1, 1])
    print '0 and 0 = %d' % and_perceptron.predict([0, 0])
    print '1 and 0 = %d' % and_perceptron.predict([1, 0])
    print '0 and 1 = %d' % and_perceptron.predict([0, 1])