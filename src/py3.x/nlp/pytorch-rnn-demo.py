# coding: utf-8

###
# Thanks to http://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
# Thanks to Ngarneau https://github.com/ngarneau/understanding-pytorch-batching-lstm
# Integrator :雪上-kia
###

from __future__ import division, print_function, unicode_literals

import glob
import random
import string
import unicodedata
from io import open

from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix)

import torch
import torch.optim as optim
from torch import nn as nn
from torch import autograd
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

random.seed(1)


def findFiles(path):
    return glob.glob(path)



# 把 Unicode 转换成 ASCII；thanks to http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn' and c in all_letters)


def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]


"""
Dataset class是 pytorch 的特色之一，能够将数据封装起来，方便以 batch 的形式训练数据
官网的教程中是一个一个(name, country)传进去的，而加入 Dataset class 是为了封装后一次性传进去 batch 个(name, country)
"""
class PaddedTensorDataset(Dataset):
    """Dataset wrapping data, target and length tensors.

    Each sample will be retrieved by indexing both tensors along the first
    dimension.

    Arguments:
        data_tensor (Tensor): contains sample data.
        target_tensor (Tensor): contains sample targets (labels).
        length (Tensor): contains sample lengths.
        raw_data (Any): The data that has been transformed into tensor, useful for debugging
    """

    def __init__(self, data_tensor, target_tensor, length_tensor, raw_data):
        assert data_tensor.size(0) == target_tensor.size(
            0) == length_tensor.size(0)
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
        self.length_tensor = length_tensor
        self.raw_data = raw_data

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[
            index], self.length_tensor[index], self.raw_data[index]

    def __len__(self):
        return self.data_tensor.size(0)


"""
A couple useful method
"""


# 把名字向量化，其实就是返回每个字母在vocabulary中的idex，比如San =（4,5,9），Snai = （4,9,5,10）   --figure 1 
def vectorize_data(data, to_ix):
    return [[to_ix[tok] if tok in to_ix else to_ix['UNK'] for tok in seq]
            for seq, y in data]


# 由于每个名字的长度不同，那它们转换成向量就有长有短，如果按照官网的方法每次只传一个名字是没有影响的
# 但是我们这里采用了一次传batch个，为了能一次性传给神经网络，就需要把所有的名字的长度都对齐batch中名字最长的那个
# 方法就是用0填充  比如Snai = （4,9,5,10）是最长的，那San就要变成 （4,5,9,0）                       --figure 2
def pad_sequences(vectorized_seqs, seq_lengths):
    seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long()
    for idx, (seq, seqlen) in enumerate(zip(vectorized_seqs, seq_lengths)):
        seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
    return seq_tensor


# target的输入必须要是long.variable而不是float
def create_dataset(data, x_to_ix, y_to_ix, bs=4):
    vectorized_seqs = vectorize_data(data, x_to_ix)
    seq_lengths = torch.LongTensor([len(s) for s in vectorized_seqs])
    seq_tensor = pad_sequences(vectorized_seqs, seq_lengths)
    target_tensor = torch.LongTensor([y_to_ix[y] for _, y in data])
    raw_data = [x for x, _ in data]
    return DataLoader(
        PaddedTensorDataset(seq_tensor, target_tensor, seq_lengths, raw_data),
        batch_size=bs)


# input此时还是B x S  (batch_size * seq_lenth)               --这个函数可以看到后面的网络结构后再回来理解
#而embedding层的输入要求是S x B  输出时S x B x I (embedding size)，故需要将input 进行 transpose
def sort_batch(batch, ys, lengths):
    seq_lengths, perm_idx = lengths.sort(0, descending=True)
    seq_tensor = batch[perm_idx]
    targ_tensor = ys[perm_idx]
    return seq_tensor.transpose(0, 1), targ_tensor, seq_lengths


#分训练，测试，内部验证（dev）
def train_dev_test_split(data):
    train_ratio = int(len(data) * 0.8)  # 80% of dataset
    train = data[:train_ratio]
    test = data[train_ratio:]
    valid_ratio = int(len(train) * 0.8)  # 20% of train set
    dev = train[valid_ratio:]
    return train, dev, test


# tag == target == label == category == country 
def build_vocab_tag_sets(data):
    vocab = set()
    tags = set()
    for name in data:
        chars = set(name[0])
        # 取并集
        vocab = vocab.union(chars)
        tags.add(name[1])
    return vocab, tags


def make_to_ix(data, to_ix=None):
    if to_ix is None:
        to_ix = dict()
    for c in data:
        to_ix[c] = len(to_ix)
    return to_ix


# 打包模型和所有的输入，最终返回预测结果和loss,要注意所有输入都要从tensor变成variable，
def apply(model, criterion, batch, targets, lengths):
    pred = model(torch.autograd.Variable(batch), lengths.cpu().numpy())
    loss = criterion(pred, torch.autograd.Variable(targets))
    return pred, loss


"""
Training and evaluation methods
"""


def train_model(model, optimizer, train, dev, x_to_ix, y_to_ix):
    criterion = nn.NLLLoss(size_average=False)
    for epoch in range(20):
        print("Epoch {}".format(epoch))
        y_true = list()
        y_pred = list()
        total_loss = 0
        for batch, targets, lengths, raw_data in create_dataset(train, x_to_ix, y_to_ix, bs=TRAIN_BATCH_SIZE):
            batch, targets, lengths = sort_batch(batch, targets, lengths)

            model.zero_grad()
            pred, loss = apply(model, criterion, batch, targets, lengths)
            loss.backward()
            optimizer.step()

            pred_idx = torch.max(pred, 1)[1]
            y_true += list(targets.int())
            y_pred += list(pred_idx.data.int())
            total_loss += loss
        acc = accuracy_score(y_true, y_pred)
        val_loss, val_acc = evaluate_validation_set(model, dev, x_to_ix,
                                                    y_to_ix, criterion)
        print(
            "Train loss: {} - acc: {} \nValidation loss: {} - acc: {}".format(
                list(total_loss.data.float())[0] / len(train), acc, val_loss,
                val_acc))
    return model


def evaluate_validation_set(model, devset, x_to_ix, y_to_ix, criterion):
    y_true = list()
    y_pred = list()
    total_loss = 0
    for batch, targets, lengths, raw_data in create_dataset(devset, x_to_ix, y_to_ix, bs=VALIDATION_BATCH_SIZE):
        batch, targets, lengths = sort_batch(batch, targets, lengths)
        pred, loss = apply(model, criterion, batch, targets, lengths)
        pred_idx = torch.max(pred, 1)[1]
        y_true += list(targets.int())
        y_pred += list(pred_idx.data.int())
        total_loss += loss
    acc = accuracy_score(y_true, y_pred)
    return list(total_loss.data.float())[0] / len(devset), acc


def evaluate_test_set(model, test, x_to_ix, y_to_ix):
    y_true = list()
    y_pred = list()

    for batch, targets, lengths, raw_data in create_dataset(
            test, x_to_ix, y_to_ix, bs=TEST_BATCH_SIZE):
        batch, targets, lengths = sort_batch(batch, targets, lengths)

        pred = model(torch.autograd.Variable(batch), lengths.cpu().numpy())
        pred_idx = torch.max(pred, 1)[1]
        y_true += list(targets.int())
        y_pred += list(pred_idx.data.int())

    print(len(y_true), len(y_pred))
    print(classification_report(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))


"""
模型是一层embedding，一层LSTM，一层全连接,全连接加一个softmax。
输入是一个batch的名字组成的向量和这个batch中最长的名字的长度即seq_lenth,也就是LSTM中所谓的time_step；
输出为一个名字对应各个国家的可能性。

在这里，embedding层的意义是避免出现太多的0值而影响结果，因为我们之前为了对齐给名字向量手动填充了很多0，
通过embedding，我们可以将其归一化（所有的值都在-2~2之间）并避免出现0值。
而实际应用中embedding层的价值更多的表现在它可以降维并能展示词与词之间关联性。      --figure 4

pack 在这里也很有趣，但是不是必须的。pack就是将我们填充后的向量全部整合在一起，一次输入就是一个完整的tensor，使得运算速度更快
                                                                                   --figure 5
还有一点这里加入了一个最大池化层max_pool处理输出，和CNN中的池化没有太多区别，在这里也不是必须的。
                                                                                   --figure 9
"""
class NamesRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_size):
        super(NamesRNN, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.char_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1)

        self.fully_connected_layer = nn.Linear(hidden_dim, output_size)
        self.softmax = nn.LogSoftmax()

    def init_hidden(self, batch):
        return (autograd.Variable(torch.randn(2, batch, self.hidden_dim)),
                autograd.Variable(torch.randn(2, batch, self.hidden_dim)))

    def _get_lstm_features(self, names, lengths):
        self.hidden = self.init_hidden(names.size(-1))
        embeds = self.char_embeds(names)  # Figure 4
        packed_input = pack_padded_sequence(embeds, lengths)  # Figure 5
        packed_output, (ht, ct) = self.lstm(packed_input, self.hidden)  # Figure 6
        lstm_out, _ = pad_packed_sequence(packed_output)  # Figure 7
        lstm_out = torch.transpose(lstm_out, 0, 1)
        lstm_out = torch.transpose(lstm_out, 1, 2)
        lstm_out = F.tanh(lstm_out)  # Figure 8
        lstm_out, indices = F.max_pool1d(lstm_out, lstm_out.size(2), return_indices=True)  # Figure 9
        lstm_out = lstm_out.squeeze(2)  #对维度的修正，使其符合输入格式
        lstm_out = F.tanh(lstm_out)
        lstm_feats = self.fully_connected_layer(lstm_out)
        output = self.softmax(lstm_feats)  # Figure 10
        return output

    def forward(self, name, lengths):
        return self._get_lstm_features(name, lengths)


"""
Method for debugging purpose
"""
def filter_for_visual_example(train):
    new_t = list()
    for x in train:
        if len(x[0]) == 6:
            new_t.append(x)
            break
    for x in train:
        if len(x[0]) == 5:
            new_t.append(x)
            break
    for x in train:
        if len(x[0]) == 4:
            new_t.append(x)
            break
    for x in train:
        if len(x[0]) == 3:
            new_t.append(x)
            break
    return new_t


def load_data(file_directory):
    # 文件名，也就是目标变量的值
    all_categories = []
    data = list()

    for filename in findFiles(file_directory):
        print("文件路径: ", filename)
        category = filename.split('/')[-1].split('.')[0]
        all_categories.append(category)

        lines = readLines(filename)
        for l in lines:
            data.append((l, category))

    data = random.sample(data, len(data))  # 打乱数据
    return all_categories, data


if __name__ == "__main__":

    """
    Vocabulary的组成与官网教程相同。
    由于所有的名字都是由ASCII码组成，此处的ascii_letters包括所有大小写字母。
    """
    TRAIN_BATCH_SIZE = 32
    VALIDATION_BATCH_SIZE = 1
    TEST_BATCH_SIZE = 1

    all_letters = string.ascii_letters + " .,;'"
    n_letters = len(all_letters)

    category_lines = {}
    # 读取所有的名字文件并将它们按照 [(name,country)...]的方式重组
    all_categories, data = load_data('/opt/data/NLP/names/*.txt')
    # 打印文件的前5行
    print(data[:5])

    # 获取所有的文本分类 + 分类的数量
    n_categories = len(all_categories)
    print("\n总共: %s 分类 \n %s" % (n_categories, all_categories))

    """
    data 按照 8:2 拆分 train 和 test(预测) 样本集合
    train 按照 8:2 拆分 train 和 dev(校验) 样本集合
    """
    train, dev, test = train_dev_test_split(data)
    # train = filter_for_visual_example(train)
    # print(train)

    vocab, tags = build_vocab_tag_sets(train)

    chars_to_idx = {'PAD': 0, 'UNK': 1}
    # 给 vocab 和 tags 中的元素加 index
    chars_to_idx = make_to_ix(sorted(list(vocab)), chars_to_idx)  # Really important to sort it if you save your model for later
    tags_to_idx = make_to_ix(sorted(list(tags)))

    model = NamesRNN(len(chars_to_idx), 128, 32, len(tags))  #voc_size, embedding_size, lstm的hidden_size, target_size
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    model = train_model(model, optimizer, train, dev, chars_to_idx, tags_to_idx)
