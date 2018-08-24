# coding: utf-8

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim


def to_scalar(var):  #var是Variable,维度是１
    # returns a python float
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):  # vec是1*5, type是Variable

    max_score = vec[0, argmax(vec)]
    # max_score维度是１，　max_score.view(1,-1)维度是１＊１，max_score.view(1, -1).expand(1, vec.size()[1])的维度是１＊５
    max_score_broadcast = max_score.view(1, -1).expand(
        1, vec.size()[1])  # vec.size()维度是1*5
    return max_score + torch.log(
        torch.sum(torch.exp(vec - max_score_broadcast)))  # 为什么指数之后再求和，而后才log呢


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

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (autograd.Variable(torch.randn(2, 1, self.hidden_dim // 2)),
                autograd.Variable(torch.randn(2, 1, self.hidden_dim // 2)))

    #预测序列的得分
    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.Tensor(1, self.tagset_size).fill_(-10000.)

        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = autograd.Variable(
            init_alphas)  # 初始状态的forward_var，随着step t变化

        # Iterate through the sentence
        for feat in feats:  # feat的维度是５
            alphas_t = []  # The forward variables at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(1, -1).expand(
                    1, self.tagset_size)  # 维度是1*5

                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)  #维度是１＊５
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                # 第一次迭代时理解：
                # trans_score所有其他标签到Ｂ标签的概率
                # 由lstm运行进入隐层再到输出层得到标签Ｂ的概率，emit_score维度是１＊５，5个值是相同的
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var))
            forward_var = torch.cat(alphas_t).view(1, -1)  # 到第（t-1）step时５个标签的各自分数
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)

        return alpha

    # 得到feats
    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        #embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        embeds = self.word_embeds(sentence)

        embeds = embeds.unsqueeze(1)

        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)

        lstm_feats = self.hidden2tag(lstm_out)

        return lstm_feats

    #得到gold_seq tag的score
    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = autograd.Variable(torch.Tensor([0]))
        tags = torch.cat([torch.LongTensor([self.tag_to_ix[START_TAG]]), tags])  # 将START_TAG的标签３拼接到tag序列上

        for i, feat in enumerate(feats):
            #self.transitions[tags[i + 1], tags[i]] 实际得到的是从标签i到标签i+1的转移概率
            #feat[tags[i+1]], feat是step i 的输出结果，有５个值，对应B, I, E, START_TAG, END_TAG, 取对应标签的值

            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[
                i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    #解码，得到预测的序列，以及预测序列的得分
    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = autograd.Variable(init_vvars)
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[
                    next_tag]  #其他标签（B,I,E,Start,End）到标签next_tag的概率
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id])
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(
                1, -1)  #从step0到step(i-1)时5个序列中每个序列的最大score
            backpointers.append(bptrs_t)  #bptrs_t有５个元素

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[
            self.tag_to_ix[STOP_TAG]]  #其他标签到STOP_TAG的转移概率
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):  #从后向前走，找到一个best路径
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()  # 把从后向前的路径正过来
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)

        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 5
HIDDEN_DIM = 4

# Make up some training data
training_data = [("the wall street journal reported today that apple corporation made money".split(), "B I I I O O O B I O O".split()),
                 ("georgia tech is a university in georgia".split(), "B I O O O O B".split())]

word_to_ix = {}
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}

model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

# Check predictions before training
# precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
# precheck_tags = torch.LongTensor([tag_to_ix[t] for t in training_data[0][1]])
# print(model(precheck_sent))

# Make sure prepare_sequence from earlier in the LSTM section is loaded
for epoch in range(1):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Variables of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = torch.LongTensor([tag_to_ix[t] for t in tags])

        # Step 3. Run our forward pass.
        neg_log_likelihood = model.neg_log_likelihood(sentence_in, targets)

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        neg_log_likelihood.backward()
        optimizer.step()


# Check predictions after training
precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
print(model(precheck_sent)[0])  # 得分
print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
print(model(precheck_sent)[1])  # tag sequence
