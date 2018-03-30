# coding: utf-8

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt


# 继承 torch 的 Model
class Net(torch.nn.Module):
    # 初始化-搭建神经网络层所需要的信息
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        # 定义每层用什么样的形式
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    # 搭建-神经网络前向传播的过程
    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

net = Net(n_feature=1, n_hidden=10, n_output=1)
print(net)


# # torch.linspace 表示在 -1和1之间等距采取100各点 
# # torch.unsqueeze 表示对老的tensor定位输出的方向，dim表示以行/列的形式输出
# x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
# y = x.pow(2) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)

# # 用 Variable 来修饰这些数据 tensor
# x, y = Variable(x), Variable(y)

# # 画图
# # scatter 打印散点图
# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()


