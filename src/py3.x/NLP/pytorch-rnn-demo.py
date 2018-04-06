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

from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)

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


"""
Vocabulary的组成与官网教程相同。
   由于所有的名字都是由ASCII码组成，此处的ascii_letters包括所有大小写字母。
"""

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)


# 把Unicode转换成ASCII；thanks to http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn' and c in all_letters)


# 读取所有的名字文件并将它们按照 [(name,country)...]的方式重组
category_lines = {}
all_categories = []


def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]


data = list()

for filename in findFiles('/opt/data/nlp/names/*.txt'):
    category = filename.split('/')[-1].split('.')[0]
    all_categories.append(category)
    lines = readLines(filename)
    for l in lines:
        data.append((l, category))

data = random.sample(data, len(data))  # 打乱数据

n_categories = len(all_categories)
print(all_categories)
# print(category_lines)
print(data[:5])

TRAIN_BATCH_SIZE = 32
VALIDATION_BATCH_SIZE = 1
TEST_BATCH_SIZE = 1
