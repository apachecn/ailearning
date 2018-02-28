#!/usr/bin/python
# encoding: utf-8

import math
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
from keras.models import load_model

# 如果使用相同的seed()值，则每次生成的随即数都相同；
numpy.random.seed(7)


def load_data():
    # 加载数据
    dataframe = read_csv('sp5001.csv', header=0, names=['价格'], usecols=[1], engine='python', skipfooter=3)
    dataset = dataframe.values
    # print('dataset=', dataset)
    # 将整型变为float
    dataset = dataset.astype('float32')
    return dataset


# X is the number of passengers at a given time (t) and Y is the number of passengers at the next time (t + 1).
# look_back 表示 时间的步长为 1; 将数组转化为矩阵
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        dataX.append(dataset[i:(i + look_back), 0])
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)


def splitData(dataset, look_back):
    # 归一化数据
    # 优化点：用源数据/还是用增长的值
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # 拆分数据集（可任意调整比例）
    train_size = int(len(dataset) * 0.8)
    train, test = dataset[0:train_size, :], dataset[train_size:, :]

    # 转化数据
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    print("dataX=%s \n dataY=%s" % (trainX.T, trainY.T))

    # reshape input to be [samples, time steps, features]
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    return dataset, scaler, trainX, trainY, testX, testY


def getModel(trainX, trainY, look_back):
    # 创建并拟合 LSTM 网络
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
    return model


def trainModel(look_back=1):
    # 加载数据
    dataset = load_data()
    # 分析数据
    # plt.plot(dataset)
    # plt.show()

    # 拆分数据机
    dataset, scaler, trainX, trainY, testX, testY = splitData(dataset, look_back)
    joblib.dump(dataset, "dataset.data")
    joblib.dump(trainX, "trainX.data")
    joblib.dump(trainY, "trainY.data")
    joblib.dump(testX, "testX.data")
    joblib.dump(testY, "testY.data")
    joblib.dump(testY, "testY.data")
    joblib.dump(scaler, "scaler.clf")

    # 获取模型
    model = getModel(trainX, trainY, look_back)
    return model


if __name__ == "__main__":

    # look_back 时间的波动区间（1天/7天/1个月/1年？？）
    look_back = 1

    # # 训练模型
    # model = trainModel(look_back)
    # # 保存模型
    # model.save('my_model.h5')
    # del model


    # 加载模型
    model = load_model('my_model.h5')
    # 加载数据
    dataset = joblib.load("dataset.data")
    trainX = joblib.load("trainX.data")
    trainY = joblib.load("trainY.data")
    testX = joblib.load("testX.data")
    testY = joblib.load("testY.data")
    scaler = joblib.load("scaler.clf")

    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])

    # # 模型评估
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
    print('Test Score: %.2f RMSE' % (testScore))

    # shift train predictions for plotting
    # 1. empty_like 返回一个和 dataset 相似的随机矩阵
    # 2. 置空
    # 3. 赋值
    trainPredictPlot = numpy.empty_like(dataset)
    trainPredictPlot[:, :] = numpy.nan
    trainPredictPlot[look_back: len(trainPredict)+look_back, :] = trainPredict

    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(dataset)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(dataset))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()
