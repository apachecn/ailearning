# *-* coding:utf-8 *-*
'''
@author: 片刻
@date: 20200901 22:02
'''

class TextNER(object):
    DEBUG = False
    path_root = "/home/wac/jiangzhonglian"
    if DEBUG:
        path_root = "/Users/jiangzl/work/data/深度学习/nlp/命名实体识别/data"

    path_train  = '%s/train_data.data'  % path_root
    path_test   = '%s/test_data.data'   % path_root
    path_config = '%s/config.pkl'       % path_root
    path_model  = '%s/model.h5'         % path_root

    # 迭代次数
    EPOCHS = 3
    # embedding的列数
    EMBED_DIM = 128
    # LSTM的列数
    BiLSTM_UNITS = 128


class Config(object):
    nlp_ner = TextNER()
