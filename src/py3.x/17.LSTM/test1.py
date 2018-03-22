#!/usr/bin/python
# encoding: utf-8

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
import matplotlib
# be able to save images on server
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# date-time parsing function for loading the dataset
def parser(x):
    return pd.datetime.strptime('190' + x, '%Y-%m')


# frame a sequence as a supervised learning problem
def timeSeries_to_supervised(data, lag=1):

    # 将 Series 转坏为 DataFrame 类型
    df = pd.DataFrame(data)
    # print('1=' * 10, data.head(5))
    # print('2=' * 10, df.head(5))

    # shift 行坐标，向下移动一位
    columns = [df.shift(i) for i in range(1, lag + 1)]
    # print('0='*10, columns)
    # print('3='*10, columns[0].head(5))
    
    '''
    # copy 一列变2列，并且第一列向下移动一位
    0	 NaN -120.1
    1  -120.1   37.2
    2	37.2  -63.8
    3   -63.8   61.0
    4	61.0  -11.8
    '''
    columns.append(df)
    df = pd.concat(columns, axis=1)
    # print('0='*10, columns)
    # print('1='*10, df)
    return df


# create a differenced pd.Series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return pd.Series(diff)


# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]


# scale train and test data to [-1, 1]
# 归一化处理，压缩到 [-1, 1] 之间
def scale(train, test):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled


# inverse scaling for a forecasted value
def invert_scale(scaler, X, yhat):
    new_row = [x for x in X] + [yhat]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]


# fit an LSTM network to training data
# n_batch
# nb_epoch
# n_neurons 神经元的个数
def fit_lstm(train, n_batch, nb_epoch, n_neurons):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    print('x=', X)
    model = Sequential()
    # https://keras.io/layers/recurrent/#lstm
    # 
    model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=n_batch, verbose=0, shuffle=False)
        model.reset_states()
    return model


# run a repeated experiment
def experiment(series, n_lag, n_repeats, n_epochs, n_batch, n_neurons):
    # transform data to be stationary

    # 获取 trainY 的值 和 前后的差值
    raw_values = series.values
    diff_values = difference(raw_values, 1)
    print('=' * 10, raw_values[:5])
    print('=' * 10, diff_values.head(5).values)

    # 将数据转化为监督学习数据，也就是时间序列的前后关系（前一天 预测 后一天数据）
    # 剔除 n_lag 之前的 None 的值
    supervised = timeSeries_to_supervised(diff_values, n_lag)
    supervised_values = supervised.values[n_lag:, :]
    print('=' * 10, supervised_values[:5])

    # 分离 训练和测试数据
    train, test = supervised_values[0:-12], supervised_values[-12:]

    # 归一化处理，压缩到 [-1, 1] 之间
    scaler, train_scaled, test_scaled = scale(train, test)
    # run experiment
    error_scores = list()
    for r in range(n_repeats):
        # fit the model
        train_trimmed = train_scaled[2:, :]
        lstm_model = fit_lstm(train_trimmed, n_batch, n_epochs, n_neurons)
        # forecast test dataset
        test_reshaped = test_scaled[:, 0:-1]
        test_reshaped = test_reshaped.reshape(len(test_reshaped), 1, 1)
        output = lstm_model.predict(test_reshaped, batch_size=n_batch)
        predictions = list()
        for i in range(len(output)):
            yhat = output[i, 0]
            X = test_scaled[i, 0:-1]
            # invert scaling
            yhat = invert_scale(scaler, X, yhat)
            # invert differencing
            yhat = inverse_difference(raw_values, yhat,
                                      len(test_scaled) + 1 - i)
            # store forecast
            predictions.append(yhat)
        # report performance
        rmse = sqrt(mean_squared_error(raw_values[-12:], predictions))
        print('%d) Test RMSE: %.3f' % (r + 1, rmse))
        error_scores.append(rmse)
    return error_scores


# configure the experiment
def run():
    # load dataset
    series = pd.read_csv(
        'shampoo-sales.csv',
        header=0,
        parse_dates=[0],
        index_col=0,
        squeeze=True,
        date_parser=parser)
    print(series.head(5))
    print(np.shape(series))

    # n_lag 表示时间窗口（上下/左右）移动的幅度
    n_lag = 1
    # 循环遍历的次数
    n_repeats = 2
    n_epochs = 1000
    n_batch = 4
    n_neurons = 3
    # run the experiment
    results = pd.DataFrame()
    results['results'] = experiment(series, n_lag, n_repeats, n_epochs,
                                    n_batch, n_neurons)
    # summarize results
    print(results.describe())
    # save boxplot
    results.boxplot()
    plt.savefig('experiment_baseline.png')

# entry point


run()
