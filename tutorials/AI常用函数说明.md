# AI常用函数说明

## numpy 相关

> from numpy import random, mat, eye

```py
'''
# NumPy 矩阵和数组的区别
NumPy存在2中不同的数据类型:
    1. 矩阵 matrix
    2. 数组 array
相似点: 
    都可以处理行列表示的数字元素
不同点: 
    1. 2个数据类型上执行相同的数据运算可能得到不同的结果。
    2. NumPy函数库中的 matrix 与 MATLAB中 matrices 等价。
'''
from numpy import random, mat, eye

# 生成一个 4*4 的随机数组
randArray = random.rand(4, 4)
# 转化关系， 数组转化为矩阵
randMat = mat(randArray)
'''
.I 表示对矩阵求逆(可以利用矩阵的初等变换)
   意义: 逆矩阵是一个判断相似性的工具。逆矩阵A与列向量p相乘后，将得到列向量q，q的第i个分量表示p与A的第i个列向量的相似度。
   参考案例链接: 
   https://www.zhihu.com/question/33258489
   http://blog.csdn.net/vernice/article/details/48506027
.T 表示对矩阵转置(行列颠倒)
    * 等同于: .transpose()
.A 返回矩阵基于的数组
    参考案例链接: 
    http://blog.csdn.net/qq403977698/article/details/47254539
'''
invRandMat = randMat.I
TraRandMat = randMat.T
ArrRandMat = randMat.A
# 输出结果
print('randArray=(%s) \n' % type(randArray), randArray)
print('randMat=(%s) \n' % type(randMat), randMat)
print('invRandMat=(%s) \n' % type(invRandMat), invRandMat)
print('TraRandMat=(%s) \n' % type(TraRandMat), TraRandMat)
print('ArrRandMat=(%s) \n' % type(ArrRandMat), ArrRandMat)
# 矩阵和逆矩阵 进行求积 (单位矩阵，对角线都为1嘛，理论上4*4的矩阵其他的都为0)
myEye = randMat*invRandMat
# 误差
print(myEye - eye(4))
```

> np.dot

```py
import numpy as np

a = np.array([2, 3])
b = np.array([4, 5])
np.dot(a, b, out=None)  #该函数的作用是获取两个元素a,b的乘积

Out[1]: 23 = 2*4 + 3*5

a = np.array([2, 3, 4])
b = np.array([5, 6, 7])
np.dot(a, b, out=None)  #该函数的作用是获取两个元素a,b的乘积

Out[2]: 56 = 2*5 + 3*6 + 4*7
```

> array sum/mean

```py
import numpy as np

# ---- sum ---- #
a = np.array([[2, 3, 4], [2, 3, 4]])

# 纵向求和: 0 表示某一列所有的行求和
a.sum(axis=0)
Out[6]: array([4, 6, 8])

# 横向求和: 1 表示某一行所有的列求和
a.sum(axis=1)
Out[7]: array([9, 9])


# ---- mean ---- #
a = np.array([[2, 3, 4], [12, 13, 14]])

# 纵向求平均: 0 表示某一列所有的行求和
a.mean(axis=0)
Out[13]: array([7., 8., 9.])

# 横向求平均: 1 表示某一行所有的列求平均
a.mean(axis=1)
Out[14]: array([ 3., 13.])

```

> np.newaxis

* numpy 添加新的维度: newaxis(可以给原数组增加一个维度)

```py
import numpy as np

In [59]: x = np.random.randint(1, 8, size=(2, 3, 4))
In [60]: y = x[:, np.newaxis, :, :]
In [61]: Z = x[ :, :, np.newaxis, :]

In [62]: x. shape
Out[62]: (2, 3, 4)

In [63]: y. shape
Out[63]: (2, 1, 3, 4)

In  [64]: z. shape
Out [64]: (2, 3, 1, 4)
```

## pandas 相关

> df.shift()

```
DataFrame.shift(periods=1, freq=None, axis=0)

* periods: 类型为int，表示移动的幅度，可以是正数，也可以是负数，默认值是1, 
    1就表示移动一次，注意这里移动的都是数据，而索引是不移动的，移动之后没有对应值的，就赋值为NaN
* freq: DateOffset, timedelta, or time rule string，可选参数，默认值为None，只适用于时间序列，如果这个参数存在，那么会按照参数值移动时间索引，而数据值没有发生变化。
* axis: {0, 1, ‘index’, ‘columns’}，表示移动的方向，如果是0或者’index’表示上下移动，如果是1或者’columns’，则会左右移动。
```

```py

"""
index	value1
A	0
B	1
C	2
D	3
"""

df.shift()  # 或 df.shift(1)
# 就会变成如下：

"""
index	value1
A	NaN
B	0
C	1
D	2
"""

df.shift(2)

"""

index	value1
A	NaN
B	NaN
C	0
D	1
"""

df.shift(-1)

"""
index	value1
A	1
B	2
C	3
D	NaN
"""

####################

"""
index	value1
2016-06-01	0
2016-06-02	1
2016-06-03	2
2016-06-04	3
"""

# 如果 index 为时间
df1.shift(periods=1,freq=datetime.timedelta(1))

"""
index	value1
2016-06-02	0
2016-06-03	1
2016-06-04	2
2016-06-05	3
"""
```



## keras 相关

> from keras.preprocessing.sequence import pad_sequences

```python
from keras.preprocessing.sequence import pad_sequences


# padding: pre(默认) 向前补充0  post 向后补充0
# truncating: 文本超过 pad_num,  pre(默认) 删除前面  post 删除后面
x = [[1,2,3,4,5]]
x_train = pad_sequences(x, maxlen=pad_num, value=0, padding='post', truncating="post")
print("--- ", x_train[0][:20])
```
