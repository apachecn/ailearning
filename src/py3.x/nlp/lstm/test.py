# coding: utf-8
# 作者: Robert Guthrie

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

lstm = nn.LSTM(3, 3)  # 输入维度是3, 输出维度也是3
inputs = [autograd.Variable(torch.randn((1, 3)))
          for _ in range(5)]  # 构造一个长度为5的序列

# 初始化隐藏状态
hidden = (autograd.Variable(torch.randn(1, 1, 3)),
          autograd.Variable(torch.randn((1, 1, 3))))
for i in inputs:
    # 将序列的元素逐个输入到LSTM
    # 经过每步操作,hidden 的值包含了隐藏状态的信息
    out, hidden = lstm(i.view(1, 1, -1), hidden)

# 另外, 我们还可以一次对整个序列进行训练. LSTM 返回的第一个值表示所有时刻的隐状态值,
# 第二个值表示最近的隐状态值 (因此下面的 "out"的最后一个值和 "hidden" 的值是一样的).
# 之所以这样设计, 是为了通过 "out" 的值来获取所有的隐状态值, 而用 "hidden" 的值来
# 进行序列的反向传播运算, 具体方式就是将它作为参数传入后面的 LSTM 网络.

# 增加额外的第二个维度
inputs = torch.cat(inputs).view(len(inputs), 1, -1)
hidden = (autograd.Variable(torch.randn(1, 1, 3)), autograd.Variable(
    torch.randn((1, 1, 3))))  # 清空输出隐状态
out, hidden = lstm(inputs, hidden)
print("out: \n", out)
print("hidden: \n", hidden)
