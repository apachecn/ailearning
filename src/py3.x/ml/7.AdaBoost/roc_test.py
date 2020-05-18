import numpy as np

from .adaboost import ada_boost_train_ds, ada_classify, load_data_set, plot_roc


def test():
    # D = np.mat(np.ones((5, 1)) / 5)
    # data_mat, class_labels = load_sim_data()
    # print(data_mat.shape)
    # result = build_stump(data_mat, class_labels, D)
    # print(result)
    # classifier_array, agg_class_est = ada_boost_train_ds(data_mat, class_labels, 9)
    # print(classifier_array, agg_class_est)
    data_mat, class_labels = load_data_set('../../../input/7.AdaBoost/horseColicTraining2.txt')
    print(data_mat.shape, len(class_labels))
    weak_class_arr, agg_class_est = ada_boost_train_ds(data_mat, class_labels, 40)
    print(weak_class_arr, '\n-----\n', agg_class_est.T)
    '''
    agg_class_est是m*1维的矩阵，需先对其转置，再执行plot_roc()
    '''
    plot_roc(agg_class_est.T, class_labels)
    data_arr_test, label_arr_test = load_data_set("../../../input/7.AdaBoost/horseColicTest2.txt")
    m = np.shape(data_arr_test)[0]
    predicting10 = ada_classify(data_arr_test, weak_class_arr)
    err_arr = np.mat(np.ones((m, 1)))
    # 测试: 计算总样本数，错误样本数，错误率
    print(m,
          err_arr[predicting10 != np.mat(label_arr_test).T].sum(),
          err_arr[predicting10 != np.mat(label_arr_test).T].sum() / m
          )
