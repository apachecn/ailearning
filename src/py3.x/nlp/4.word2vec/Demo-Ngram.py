# coding: utf-8

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# word_to_ix = {"hello": 0, "world": 1}
# # 2表示有2个词，5表示5维度，其实也就是一个2x5的矩阵
# embeds = nn.Embedding(2, 5)
# lookup_tensor = torch.LongTensor([word_to_ix["hello"]])
# hello_embed = embeds(autograd.Variable(lookup_tensor))
# print(hello_embed)


CONTEXT_SIZE = 2
EMBEDDING_DIM = 10

# 我们将使用 Shakespeare Sonnet 2
test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()

# 建造一系列元组. 每个元组 ([word_i-2, word_i-1] => 特征, word_i => 目标变量)
# 目标词的条件概率只与其之前的 n 个词有关
trigrams = [([test_sentence[i], test_sentence[i + 1]], test_sentence[i + 2])
            for i in range(len(test_sentence) - 2)]
# 打印前3行
print("打印数据前3行: \n", trigrams[:3])

# 词集选择， enumerate是带序列号的增序ID列表
vocab = set(test_sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)}


class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        # embedding_dim 设置 Embedding 的维度大小
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)


losses = []
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)


for epoch in range(10):
    total_loss = torch.Tensor([0])
    for context, target in trigrams:

        # 1
        context_idxs = [word_to_ix[w] for w in context]
        context_var = autograd.Variable(torch.LongTensor(context_idxs))

        # 2
        model.zero_grad()

        # 3
        log_probs = model(context_var)

        # 4
        loss = loss_function(log_probs, autograd.Variable(torch.LongTensor([word_to_ix[target]])))

        # 5
        loss.backward()
        optimizer.step()

        total_loss += loss.data
    losses.append(total_loss)
print("计算损失函数: \n", losses)  # The loss decreased every iteration over the training data!
