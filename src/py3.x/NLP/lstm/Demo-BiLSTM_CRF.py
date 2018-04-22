# coding: utf-8
# 作者: Robert Guthrie

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)


def to_scalar(var):
    # 返回 python 浮点数 (float)
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    # 以 python 整数的形式返回 argmax
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)


# 使用数值上稳定的方法为前向算法计算指数和的对数
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)

        # 将LSTM的输出映射到标记空间
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # 过渡参数矩阵. 条目 i,j 是
        # *从 * j *到* i 的过渡的分数
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # 这两句声明强制约束了我们不能
        # 向开始标记标注传递和从结束标注传递
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (autograd.Variable(torch.randn(2, 1, self.hidden_dim // 2)),
                autograd.Variable(torch.randn(2, 1, self.hidden_dim // 2)))

    def _forward_alg(self, feats):
        # 执行前向算法来计算分割函数
        init_alphas = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        # START_TAG 包含所有的分数
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # 将其包在一个变量类型中继而得到自动的反向传播
        forward_var = autograd.Variable(init_alphas)

        # 在句子中迭代
        for feat in feats:
            alphas_t = []  # 在这个时间步的前向变量
            for next_tag in range(self.tagset_size):
                # 对 emission 得分执行广播机制: 它总是相同的,
                # 不论前一个标注如何
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # trans_score 第 i 个条目是
                # 从i过渡到 next_tag 的分数
                trans_score = self.transitions[next_tag].view(1, -1)
                # next_tag_var 第 i 个条目是在我们执行 对数-求和-指数 前
                # 边缘的值 (i -> next_tag)
                next_tag_var = forward_var + trans_score + emit_score
                # 这个标注的前向变量是
                # 对所有的分数执行 对数-求和-指数
                alphas_t.append(log_sum_exp(next_tag_var))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # 给出标记序列的分数
        score = autograd.Variable(torch.Tensor([0]))
        tags = torch.cat([torch.LongTensor([self.tag_to_ix[START_TAG]]), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # 在对数空间中初始化维特比变量
        init_vvars = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # 在第 i 步的 forward_var 存放第 i-1 步的维特比变量
        forward_var = autograd.Variable(init_vvars)
        for feat in feats:
            bptrs_t = []  # 存放这一步的后指针
            viterbivars_t = []  # 存放这一步的维特比变量

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] 存放先前一步标注i的
                # 维特比变量, 加上了从标注 i 到 next_tag 的过渡
                # 的分数
                # 我们在这里并没有将 emission 分数包含进来, 因为
                # 最大值并不依赖于它们(我们在下面对它们进行的是相加)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id])
            # 现在将所有 emission 得分相加, 将 forward_var
            # 赋值到我们刚刚计算出来的维特比变量集合
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # 过渡到 STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # 跟着后指针去解码最佳路径
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # 弹出开始的标签 (我们并不希望把这个返回到调用函数)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # 健全性检查
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # 不要把这和上面的 _forward_alg 混淆
        # 得到 BiLSTM 输出分数
        lstm_feats = self._get_lstm_features(sentence)

        # 给定特征, 找到最好的路径
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 5
HIDDEN_DIM = 4

# 编造一些训练数据
training_data = [(
    "the wall street journal reported today that apple corporation made money".
    split(), "B I I I O O O B I O O".split()),
                 ("georgia tech is a university in georgia".split(),
                  "B I O O O O B".split())]

word_to_ix = {}
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}

model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

# 在训练之前检查预测结果
precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
precheck_tags = torch.LongTensor([tag_to_ix[t] for t in training_data[0][1]])
print(model(precheck_sent))

# 确认从之前的 LSTM 部分的 prepare_sequence 被加载了
for epoch in range(300):  # 又一次, 正常情况下你不会训练300个 epoch, 这只是示例数据
    for sentence, tags in training_data:
        # 第一步: 需要记住的是Pytorch会累积梯度
        # 我们需要在每次实例之前把它们清除
        model.zero_grad()

        # 第二步: 为我们的网络准备好输入, 即
        # 把它们转变成单词索引变量 (Variables)
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = torch.LongTensor([tag_to_ix[t] for t in tags])

        # 第三步: 运行前向传递.
        neg_log_likelihood = model.neg_log_likelihood(sentence_in, targets)

        # 第四步: 计算损失, 梯度以及
        # 使用 optimizer.step() 来更新参数
        neg_log_likelihood.backward()
        optimizer.step()

# 在训练之后检查预测结果
precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
print(model(precheck_sent))
# 我们完成了!
