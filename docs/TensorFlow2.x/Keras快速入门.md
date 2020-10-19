# Keras 快速入门

* 安装: `pip install keras`

> Keras 发展生态支持

* 1.Keras 的开发主要由谷歌支持，Keras API 以 tf.keras 的形式包装在 TensorFlow 中。
* 2.微软维护着 Keras 的 CNTK 后端。
* 3.亚马逊 AWS 正在开发 MXNet 支持。
* 4.其他提供支持的公司包括 NVIDIA、优步、苹果（通过 CoreML）等。

## Keras dataset 生产数据

```py
# With Numpy arrays
data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

model.evaluate(data, labels, batch_size=32)

# With a Dataset
dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32)

model.evaluate(dataset)
```


## Keras Sequential 顺序模型

顺序模型是多个网络层的线性堆叠，目前支持2种方式

### 构造模型

> 1.构造器: 构建 Sequential 模型

```py
from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential([
    Dense(32, input_shape=(784,)),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])
```

> 2.add(): 构建 Sequential 模型

```py
from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential()
model.add(Dense(32, input_dim=784))
model.add(Activation('relu'))
model.add(Dense(10)
model.add(Activation('softmax'))
```

### Dense: <https://keras.io/zh/layers/core>

Dense 指的是配置全连接层

```py
keras.layers.Dense(units, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)

# 参数
* units: 正整数，输出空间维度。
* activation: 激活函数 (详见 activations)。 若不指定，则不使用激活函数 (即，「线性」激活: a(x) = x)。
* use_bias: 布尔值，该层是否使用偏置向量。
* kernel_initializer: kernel 权值矩阵的初始化器 (详见 initializers)。
* bias_initializer: 偏置向量的初始化器 (see initializers).
* kernel_regularizer: 运用到 kernel 权值矩阵的正则化函数 (详见 regularizer)。
* bias_regularizer: 运用到偏置向的的正则化函数 (详见 regularizer)。
* activity_regularizer: 运用到层的输出的正则化函数 (它的 "activation")。 (详见 regularizer)。
* kernel_constraint: 运用到 kernel 权值矩阵的约束函数 (详见 constraints)。
* bias_constraint: 运用到偏置向量的约束函数 (详见 constraints)。
```

例如:

```py
# 作为 Sequential 模型的第一层
model = Sequential()
# 现在模型就会以尺寸为 (*, 16) 的数组作为输入，
# 其输出数组的尺寸为 (*, 32)
model.add(Dense(32, input_shape=(16,)))

# 在第一层之后，你就不再需要指定输入的尺寸了: 
model.add(Dense(32))
```

### Activation 激活函数: <https://keras-cn.readthedocs.io/en/latest/other/activations>

> 激活函数: 将线性问题变成非线性（回归问题变为分类问题），简单计算难度和复杂性。



sigmoid

hard_sigmoid

tanh

relu

softmax: 对输入数据的最后一维进行softmax，输入数据应形如(nb_samples, nb_timesteps, nb_dims)或(nb_samples,nb_dims)

elu

selu: 可伸缩的指数线性单元（Scaled Exponential Linear Unit），参考Self-Normalizing Neural Networks

softplus

softsign

linear



## Keras compile 过程

```py
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test,  y_test, verbose=2)
```

### 优化器 optimizer: <https://keras.io/optimizers>


> SGD


```py
keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False)
```

例如:

```py
from keras import optimizers
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)
```


> RMSprop

```py
keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
```

> Adagrad

```py
keras.optimizers.Adagrad(learning_rate=0.01)
```


> Adadelta

```py
keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95)
```

> Adam

```py
keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
```


> Adamax

```py
keras.optimizers.Adamax(learning_rate=0.002, beta_1=0.9, beta_2=0.999)
```


> Nadam

```py
keras.optimizers.Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999)
```


### 损失函数 loss: <https://keras.io/losses>


#### 回归

> mean_squared_error

```py
keras.losses.mean_squared_error(y_true, y_pred)
```

#### 二分类

> binary_crossentropy

```py
keras.losses.binary_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0)
```


#### 多分类

> categorical_crossentropy

```py
keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0)
```

### 评价函数 metrics: <https://keras.io/zh/metrics>

评价函数: 用来衡量`真实值`和`预测值`的差异

> binary_accuracy

对二分类问题,计算在所有预测值上的平均正确率

```py
binary_accuracy(y_true, y_pred)
```

> categorical_accuracy

对多分类问题,计算再所有预测值上的平均正确率

```py
categorical_accuracy(y_true, y_pred)
```

> sparse_categorical_accuracy

与categorical_accuracy相同,在对稀疏的目标值预测时有用

```py
sparse_categorical_accuracy(y_true, y_pred)
```

> top_k_categorical_accuracy

计算top-k正确率,当预测值的前k个值中存在目标类别即认为预测正确

```py
top_k_categorical_accuracy(y_true, y_pred, k=5)
```

> sparse_top_k_categorical_accuracy

与top_k_categorical_accracy作用相同，但适用于稀疏情况

```py
sparse_top_k_categorical_accuracy(y_true, y_pred, k=5)
```


## Keras load/save 模型持久化

> 保存模型

```py
import tensorflow as tf

# Save entire model to a HDF5 file
model.save('my_model.h5')

# Recreate the exact same model, including weights and optimizer.
model = tf.keras.models.load_model('my_model.h5')
```

> 仅保存权重值

```py
import tensorflow as tf

# Save weights to a TensorFlow Checkpoint file
model.save_weights('./weights/my_model')
# Restore the model's state,
# this requires a model with the same architecture.
model.load_weights('./weights/my_model')


# Save weights to a HDF5 file
model.save_weights('my_model.h5', save_format='h5')
# Restore the model's state
model.load_weights('my_model.h5')
```

> 仅保存模型配置

```py
import tensorflow as tf

# Serialize a model to json format
json_string = model.to_json()
fresh_model = tf.keras.models.model_from_json(json_string)

# Serialize a model to yaml format
yaml_string = model.to_yaml()
fresh_model = tf.keras.models.model_from_yaml(yaml_string)
```

--- 

补充损失函数

> mean_absolute_error

```py
keras.losses.mean_absolute_error(y_true, y_pred)
```

> mean_absolute_percentage_error

```py
keras.losses.mean_absolute_percentage_error(y_true, y_pred)
```

> mean_squared_logarithmic_error

```py
keras.losses.mean_squared_logarithmic_error(y_true, y_pred)
```

> squared_hinge

```
keras.losses.squared_hinge(y_true, y_pred)

```

> hinge

```py
keras.losses.hinge(y_true, y_pred)
```

> categorical_hinge

```py
keras.losses.categorical_hinge(y_true, y_pred)
```

> logcosh

```py
keras.losses.logcosh(y_true, y_pred)
```


> huber_loss

```py
keras.losses.huber_loss(y_true, y_pred, delta=1.0)
```


> sparse_categorical_crossentropy

```py
keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=False, axis=-1)
```


> kullback_leibler_divergence

```py
keras.losses.kullback_leibler_divergence(y_true, y_pred)
```

> poisson

```py
keras.losses.poisson(y_true, y_pred)
```

> cosine_proximity

```py
keras.losses.cosine_proximity(y_true, y_pred, axis=-1)
```

> is_categorical_crossentropy

```py
keras.losses.is_categorical_crossentropy(loss)
```




