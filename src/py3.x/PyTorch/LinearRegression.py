# coding: utf-8

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt


# torch.linspace 表示在 -1和1之间等距采取100各点 
# torch.unsqueeze 表示对老的tensor定位输出的方向，dim表示以行/列的形式输出
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)

# # 用 Variable 来修饰这些数据 tensor
x, y = Variable(x), Variable(y)

# # 画图
# # scatter 打印散点图
# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()


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
        # 隐藏层-输出结果（输入加权、激活输出）
        x_h_o = F.relu(self.hidden(x))
        # 输出层-预测结果（输入加权）
        y = self.predict(x_h_o)
        return y


net = Net(n_feature=1, n_hidden=10, n_output=1)
print(net)


# optimizer 优化器 是训练的工具，lr表示学习率
'''
>>> # lr = 0.05     if epoch < 30
>>> # lr = 0.005    if 30 <= epoch < 80
>>> # lr = 0.0005   if epoch >= 80
'''
# 传入 net 的所有参数, 学习率（例如：学习率<=1, 如果值过高，速度会很快，但容易忽视知识点）
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
# 损失函数
loss_func = torch.nn.MSELoss()      # 预测值和真实值的误差计算公式 (均方误差)

plt.ion()   # 画图

for t in range(100):
    prediction = net(x)     # 喂给 net 训练数据 x, 输出预测值
    # 均方误差
    loss = loss_func(prediction, y)     # 计算两者的误差

    optimizer.zero_grad()   # 清空上一步的残余更新参数值
    loss.backward()         # 误差反向传播, 计算参数更新值
    optimizer.step()        # 将参数更新值施加到 net 的 parameters 上

    # 接着上面来
    if t % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data[0], fontdict={'size': 20, 'color':  'red'})
        # 刷新频率
        plt.pause(0.1)

plt.ioff()
plt.show()
