# coding: utf-8

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)


data = [("me gusta comer en la cafeteria".split(), "SPANISH"),
        ("Give it to me".split(), "ENGLISH"),
        ("No creo que sea una buena idea".split(), "SPANISH"),
        ("No it is not a good idea to get lost at sea".split(), "ENGLISH")]

test_data = [("Yo creo que si".split(), "SPANISH"),
             ("it is lost on me".split(), "ENGLISH")]

label_to_ix = {"SPANISH": 0, "ENGLISH": 1}


# word_to_ix 将词汇表中的每个单词映射到一个唯一的整数，这将成为单词向量袋中的索引
word_to_ix = {}
for sent, _ in data + test_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
print("打印词汇表: \n", word_to_ix)

# 输入的特征数
VOCAB_SIZE = len(word_to_ix)
# 输出的结果
NUM_LABELS = 2


# 继承 nn.Module
class BoWClassifier(nn.Module):

    def __init__(self, num_labels, vocab_size):
        # 调用 nn.Module 的 init 函数。 不要被语法混淆，只是总是在 nn.Module 中做
        super(BoWClassifier, self).__init__()

        # 定义您将需要的参数。
        # 在这种情况下，我们需要A和b，仿射映射的参数。
        # Torch定义了nn.Linear()，它提供了仿射图。
        # 确保你明白为什么输入维度是 vocab_size，输出是 num_labels！
        self.linear = nn.Linear(vocab_size, num_labels)

        # NOTE! 非线性日志softmax没有参数！ 所以我们不需要为此担心

    def forward(self, bow_vec):
        # 通过线性层传递输入，
        # 然后通过 log_softmax 传递。（使用对数形式的 softmax 函数）
        # 许多非线性和其他功能在 torch.nn.functional 中
        # SVM只选自己喜欢的男神，Softmax把所有备胎全部拉出来评分，最后还归一化一下
        return F.log_softmax(self.linear(bow_vec), dim=1)


def make_bow_vector(sentence, word_to_ix):
    vec = torch.zeros(len(word_to_ix))
    for word in sentence:
        vec[word_to_ix[word]] += 1
    # 返回一个有相同数据但大小不同的新的 tensor.（ -1表示其他维度 ）
    """
    # -1 表示自己不确定，让系统来计算
    y = x.view(4, 2)
    print y

    # 输出如下
    1.5600 -1.6180
    -2.0366  2.7115
    0.8415 -1.0103
    -0.4793  1.5734
    [torch.FloatTensor of size 4x2]
    """
    return vec.view(1, -1)


def make_target(label, label_to_ix):
    return torch.LongTensor([label_to_ix[label]])


model = BoWClassifier(NUM_LABELS, VOCAB_SIZE)

# 模型知道它的参数。 下面的第一个输出是A，第二个输出是b。
# 只要你在一个模块的 __init__ 函数中将一个组件分配给一个类变量，这是通过该行完成的
# self.linear = nn.Linear（...）
# 然后通过 Pytorch 开发者的一些 Python 魔法，你的模块（在这种情况下，BoWClassifier）将存储关于 nn.Linear 参数的知识
# for param in model.parameters():
#     print("\n parameters 参数有: \n", param)

# 要运行该模型，传入一个BoW variable，但包裹在一个 autograd.Variable 中
sample = data[0]
# 将文本转化为 Variable 对象
bow_vector = make_bow_vector(sample[0], word_to_ix)
print("bow_vector: \n", bow_vector)
print("autograd: \n", autograd.Variable(bow_vector))
log_probs = model(autograd.Variable(bow_vector))
# 将原始数据从 x ⇒ log (x)，无疑会将原始数据的值域进行一定的收缩。
"""lable目标变量的：最终的分类下所有结果的概率（下面为: SPANISH 和 ENGLISH 的概率分布）
Variable containing:
-0.8195 -0.5810
"""
print(log_probs)


# Run on test data before we train, just to see a before-and-after
for instance, label in test_data:
    bow_vec = autograd.Variable(make_bow_vector(instance, word_to_ix))
    log_probs = model(bow_vec)
    print(log_probs)

# 打印对应于 "creo" 的矩阵列
print('model.parameters(): \n', next(model.parameters())[:, word_to_ix["creo"]])

loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Usually you want to pass over the training data several times.
# 100 is much bigger than on a real data set, but real datasets have more than
# two instances.  Usually, somewhere between 5 and 30 epochs is reasonable.
for epoch in range(100):
    for instance, label in data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Make our BOW vector and also we must wrap the target in a
        # Variable as an integer. For example, if the target is SPANISH, then
        # we wrap the integer 0. The loss function then knows that the 0th
        # element of the log probabilities is the log probability
        # corresponding to SPANISH
        bow_vec = autograd.Variable(make_bow_vector(instance, word_to_ix))
        target = autograd.Variable(make_target(label, label_to_ix))

        # Step 3. Run our forward pass.
        log_probs = model(bow_vec)

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss = loss_function(log_probs, target)
        loss.backward()
        optimizer.step()

for instance, label in test_data:
    bow_vec = autograd.Variable(make_bow_vector(instance, word_to_ix))
    log_probs = model(bow_vec)
    print(log_probs)

# Index corresponding to Spanish goes up, English goes down!
print(next(model.parameters())[:, word_to_ix["creo"]])
