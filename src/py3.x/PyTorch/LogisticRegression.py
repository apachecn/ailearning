# coding: utf-8

import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F  # 激励函数都在这
from torch.autograd import Variable

# 假数据
n_data = torch.ones(100, 2)  # 数据的基本形态 shape=(100, 2) 数值都为1
x0 = torch.normal(2 * n_data, 1)  # 类型0 x data (tensor), shape=(100, 2)
y0 = torch.zeros(100)  # 类型0 y data (tensor), shape=(100, 1)
x1 = torch.normal(-2 * n_data, 1)  # 类型1 x data (tensor), shape=(100, 2)
y1 = torch.ones(100)  # 类型1 y data (tensor), shape=(100, 1)

# 注意 x, y 数据的数据形式是一定要像下面一样 (torch.cat 是在合并数据, 默认是0，0表示追加行)
x = torch.cat(
    (x0, x1), 0).type(torch.FloatTensor)  # FloatTensor = 32-bit floating
y = torch.cat((y0, y1), ).type(torch.LongTensor)  # LongTensor = 64-bit integer

# torch 只能在 Variable 上训练, 所以把它们变成 Variable
x, y = Variable(x), Variable(y)

print(zip(x0, y0))
print('00', 20)
print(zip(x1, y1))

# # 画散点图
# # s表示size大小
# # c表示颜色，可以用于不同组之间的颜色
# # lw表示标记边缘的线宽。注意：默认的边框颜色 是'face'
# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
# plt.show()


# 创建神经网络的方式1：可以手动的来定义激励函数
class Net(torch.nn.Module):  # 继承 torch 的 Module
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()  # 继承 __init__ 功能
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # 隐藏层线性输出
        self.out = torch.nn.Linear(n_hidden, n_output)  # 输出层线性输出

    def forward(self, x):
        # 正向传播输入值, 神经网络分析出输出值
        x_h_o = F.relu(self.hidden(x))  # 激励函数(隐藏层的线性值)
        y = self.out(x_h_o)  # 输出值, 但是这个不是预测值, 预测值还需要再另外计算
        return y


net1 = Net(n_feature=2, n_hidden=10, n_output=2)  # 几个类别就几个 output

# # 创建神经网络的方式2：它定义好了激励函数
# net2 = torch.nn.Sequential(
#     torch.nn.Linear(1, 10),
#     torch.nn.ReLU(),
#     torch.nn.Linear(10, 1)
# )

# # net1 和 net2 效果是相同的
print('net1:\n', net1)  # net1 的结构
# print('net2:\n', net2)  # net2 的结构
# """
# net1:
#  Net (
#   (hidden): Linear (2 -> 10)
#   (out): Linear (10 -> 2)
# )
# net2:
#  Sequential (
#   (0): Linear (1 -> 10)
#   (1): ReLU ()
#   (2): Linear (10 -> 1)
# )
# """

# optimizer 优化器 是训练的工具，lr表示学习率
'''
>>> # lr = 0.05     if epoch < 30
>>> # lr = 0.005    if 30 <= epoch < 80
>>> # lr = 0.0005   if epoch >= 80
'''
optimizer = torch.optim.SGD(net1.parameters(), lr=0.02)  # 传入 net 的所有参数, 学习率
# 算误差的时候, 注意真实值!不是! one-hot 形式的, 而是1D Tensor, (batch,)
# 但是预测值是2D tensor (batch, n_classes)
loss_func = torch.nn.CrossEntropyLoss()

plt.ion()   # 画图

for t in range(100):
    out = net1(x)     # 喂给 net 训练数据 x, 输出分析值
    # 交叉熵
    loss = loss_func(out, y)     # 计算两者的误差

    optimizer.zero_grad()   # 清空上一步的残余更新参数值（也就是说：清除所有优化变量的梯度。）
    loss.backward()         # 误差反向传播, 计算参数更新值
    optimizer.step()        # 将参数更新值施加到 net 的 parameters 上

    # 接着上面来
    if t % 2 == 0:
        plt.cla()
        # 过了一道 softmax 的激励函数后的最大概率才是预测值
        prediction = torch.max(F.softmax(out), 1)[1]
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y)/200  # 预测中有多少和真实值一样
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

plt.ioff()  # 停止画图
plt.show()
