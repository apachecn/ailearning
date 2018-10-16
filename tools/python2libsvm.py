#!/usr/bin/python
# coding:utf8

from __future__ import print_function
import os
import sklearn.datasets as datasets


def get_data(file_input, separator='\t'):
    if 'libsvm' not in file_input:
        file_input = other2libsvm(file_input, separator)
    data = datasets.load_svmlight_file(file_input)
    return data[0], data[1]


def other2libsvm(file_name, separator='\t'):

    libsvm_name = file_name.replace('.txt', '.libsvm_tmp')
    libsvm_data = open(libsvm_name, 'w')

    file_data = open(file_name, 'r')
    for line in file_data.readlines():
        features = line.strip().split(separator)
        # print len(features)
        class_data = features[-1]
        svm_format = ''
        for i in range(len(features)-1):
            svm_format += " %d:%s" % (i+1, features[i])
            # print svm_format
        svm_format = "%s%s\n" % (class_data, svm_format)
        # print svm_format
        libsvm_data.write(svm_format)
    file_data.close()

    libsvm_data.close()
    return libsvm_name


def dump_data(x, y, file_output):
    datasets.dump_svmlight_file(x, y, file_output)
    os.remove("%s_tmp" % file_output)


if __name__ == "__main__":
    file_input = "db/7.AdaBoost/horseColicTest2.txt"
    file_output = "db/7.AdaBoost/horseColicTest2.libsvm"

    # 获取数据集
    x, y = get_data(file_input, separator='\t')
    print(x[3, :])
    print(y)
    # 导出数据为 libsvm
    dump_data(x, y, file_output)
