# Python 数据分析中文笔记

> 版本：0.0.1
> 作者：李金
> 邮件：[[email protected]](/cdn-cgi/l/email-protection)
> 微信：lijinwithyou

`Github` 加载 `.ipynb` 的速度较慢，建议在 [Nbviewer](http://nbviewer.ipython.org/github/lijin-THU/notes-python/blob/master/index.ipynb) 中查看该项目。

* * *

## 简介

大部分内容来自网络。

默认安装了 `Python 2.7`，以及相关的第三方包 `ipython`， `numpy`， `scipy`，`pandas`。

> life is short. use python.

推荐使用 [Anaconda](http://www.continuum.io/downloads)，这个IDE集成了大部分常用的包。

笔记内容使用 `ipython notebook` 来展示。

安装好 `Python` 和相应的包之后，可以在命令行下输入：

```
$ ipython notebook
```

来进入 `ipython notebook`。

* * *

## 基本环境配置

*   安装 [Anaconda](http://www.continuum.io/downloads) 或者 [Miniconda](http://conda.pydata.org/miniconda.html)

*   更新环境

    ```
    conda update conda
    conda update anaconda
    ```

* * *

## 参考

*   [Enthought Training on Demand](https://training.enthought.com/)
*   [Computational Statistics in Python](http://people.duke.edu/~ccc14/sta-663/index.html#rd)
*   [Scipy.org](http://scipy.org/)
*   [Deep Learning Tutorials](http://deeplearning.net/tutorial/)
*   [High Performance Scientific Computing](http://faculty.washington.edu/rjl/uwhpsc-coursera/index.html)
*   [Scipy Lectures](http://www.scipy-lectures.org/)
*   [Pandas.org](http://pandas.pydata.org/pandas-docs/stable/index.html)

* * *

## 目录

可以在 Notebook 中打开 `generate static files.ipynb`，或者命令行中运行代码 `generate_static_files.py` 来生成静态的 HTML 文件。

* * *

*   [01\. **Python 工具**](01\. python tools)
    *   [01.01 Python 简介](01\. python tools/01.01 python overview.ipynb)
    *   [01.02 Ipython 解释器](01\. python tools/01.02 ipython interpreter.ipynb)
    *   [01.03 Ipython notebook](01\. python tools/01.03 ipython notebook.ipynb)
    *   [01.04 使用 Anaconda](01\. python tools/01.04 use anaconda.ipynb)
*   [02\. **Python 基础**](02\. python essentials)
    *   [02.01 Python 入门演示](02\. python essentials/02.01 a tour of python.ipynb)
    *   [02.02 Python 数据类型](02\. python essentials/02.02 python data types.ipynb)
    *   [02.03 数字](02\. python essentials/02.03 numbers.ipynb)
    *   [02.04 字符串](02\. python essentials/02.04 strings.ipynb)
    *   [02.05 索引和分片](02\. python essentials/02.05 indexing and slicing.ipynb)
    *   [02.06 列表](02\. python essentials/02.06 lists.ipynb)
    *   [02.07 可变和不可变类型](02\. python essentials/02.07 mutable and immutable data types.ipynb)
    *   [02.08 元组](02\. python essentials/02.08 tuples.ipynb)
    *   [02.09 列表与元组的速度比较](02\. python essentials/02.09 speed comparison between list & tuple.ipynb)
    *   [02.10 字典](02\. python essentials/02.10 dictionaries.ipynb)
    *   [02.11 集合](02\. python essentials/02.11 sets.ipynb)
    *   [02.12 不可变集合](02\. python essentials/02.12 frozen sets.ipynb)
    *   [02.13 Python 赋值机制](02\. python essentials/02.13 how python assignment works.ipynb)
    *   [02.14 判断语句](02\. python essentials/02.14 if statement.ipynb)
    *   [02.15 循环](02\. python essentials/02.15 loops.ipynb)
    *   [02.16 列表推导式](02\. python essentials/02.16 list comprehension.ipynb)
    *   [02.17 函数](02\. python essentials/02.17 functions.ipynb)
    *   [02.18 模块和包](02\. python essentials/02.18 modules and packages.ipynb)
    *   [02.19 异常](02\. python essentials/02.19 exceptions.ipynb)
    *   [02.20 警告](02\. python essentials/02.20 warnings.ipynb)
    *   [02.21 文件读写](02\. python essentials/02.21 file IO.ipynb)
*   [03\. **Numpy**](03\. numpy)
    *   [03.01 Numpy 简介](03\. numpy/03.01 numpy overview.ipynb)
    *   [03.02 Matplotlib 基础](03\. numpy/03.02 matplotlib basics.ipynb)
    *   [03.03 Numpy 数组及其索引](03\. numpy/03.03 numpy arrays.ipynb)
    *   [03.04 数组类型](03\. numpy/03.04 array types.ipynb)
    *   [03.05 数组方法](03\. numpy/03.05 array calculation method.ipynb)
    *   [03.06 数组排序](03\. numpy/03.06 sorting numpy arrays.ipynb)
    *   [03.07 数组形状](03\. numpy/03.07 array shapes.ipynb)
    *   [03.08 对角线](03\. numpy/03.08 diagonals.ipynb)
    *   [03.09 数组与字符串的转换](03\. numpy/03.09 data to & from string.ipynb)
    *   [03.10 数组属性方法总结](03\. numpy/03.10 array attribute & method overview .ipynb)
    *   [03.11 生成数组的函数](03\. numpy/03.11 array creation functions.ipynb)
    *   [03.12 矩阵](03\. numpy/03.12 matrix object.ipynb)
    *   [03.13 一般函数](03\. numpy/03.13 general functions.ipynb)
    *   [03.14 向量化函数](03\. numpy/03.14 vectorizing functions.ipynb)
    *   [03.15 二元运算](03\. numpy/03.15 binary operators.ipynb)
    *   [03.16 ufunc 对象](03\. numpy/03.16 universal functions.ipynb)
    *   [03.17 choose 函数实现条件筛选](03\. numpy/03.17 choose.ipynb)
    *   [03.18 数组广播机制](03\. numpy/03.18 array broadcasting.ipynb)
    *   [03.19 数组读写](03\. numpy/03.19 reading and writing arrays.ipynb)
    *   [03.20 结构化数组](03\. numpy/03.20 structured arrays.ipynb)
    *   [03.21 记录数组](03\. numpy/03.21 record arrays.ipynb)
    *   [03.22 内存映射](03\. numpy/03.22 memory maps.ipynb)
    *   [03.23 从 Matlab 到 Numpy](03\. numpy/03.23 from matlab to numpy.ipynb)
*   [04\. **Scipy**](04\. scipy)
    *   [04.01 SCIentific PYthon 简介](04\. scipy/04.01 scienticfic python overview.ipynb)
    *   [04.02 插值](04\. scipy/04.02 interpolation with scipy.ipynb)
    *   [04.03 概率统计方法](04\. scipy/04.03 statistics with scipy.ipynb)
    *   [04.04 曲线拟合](04\. scipy/04.04 curve fitting.ipynb)
    *   [04.05 最小化函数](04\. scipy/04.05 minimization in python.ipynb)
    *   [04.06 积分](04\. scipy/04.06 integration in python.ipynb)
    *   [04.07 解微分方程](04\. scipy/04.07 ODEs.ipynb)
    *   [04.08 稀疏矩阵](04\. scipy/04.08 sparse matrix.ipynb)
    *   [04.09 线性代数](04\. scipy/04.09 linear algbra.ipynb)
    *   [04.10 稀疏矩阵的线性代数](04\. scipy/04.10 sparse linear algebra.ipynb)
*   [05\. **Python 进阶**](05\. advanced python)
    *   [05.01 sys 模块简介](05\. advanced python/05.01 overview of the sys module.ipynb)
    *   [05.02 与操作系统进行交互：os 模块](05\. advanced python/05.02 interacting with the OS - os.ipynb)
    *   [05.03 CSV 文件和 csv 模块](05\. advanced python/05.03 comma separated values.ipynb)
    *   [05.04 正则表达式和 re 模块](05\. advanced python/05.04 regular expression.ipynb)
    *   [05.05 datetime 模块](05\. advanced python/05.05 datetime.ipynb)
    *   [05.06 SQL 数据库](05\. advanced python/05.06 sql databases.ipynb)
    *   [05.07 对象关系映射](05\. advanced python/05.07 object-relational mappers.ipynb)
    *   [05.08 函数进阶：参数传递，高阶函数，lambda 匿名函数，global 变量，递归](05\. advanced python/05.08 functions.ipynb)
    *   [05.09 迭代器](05\. advanced python/05.09 iterators.ipynb)
    *   [05.10 生成器](05\. advanced python/05.10 generators.ipynb)
    *   [05.11 with 语句和上下文管理器](05\. advanced python/05.11 context managers and the with statement.ipynb)
    *   [05.12 修饰符](05\. advanced python/05.12 decorators.ipynb)
    *   [05.13 修饰符的使用](05\. advanced python/05.13 decorator usage.ipynb)
    *   [05.14 operator, functools, itertools, toolz, fn, funcy 模块](05\. advanced python/05.14 the operator functools itertools toolz fn funcy module.ipynb)
    *   [05.15 作用域](05\. advanced python/05.15 scope.ipynb)
    *   [05.16 动态编译](05\. advanced python/05.16 dynamic code execution.ipynb)
*   [06\. **Matplotlib**](06\. matplotlib)
    *   [06.01 Pyplot 教程](06\. matplotlib/06.01 pyplot tutorial.ipynb)
    *   [06.02 使用 style 来配置 pyplot 风格](06\. matplotlib/06.02 customizing plots with style sheets.ipynb)
    *   [06.03 处理文本（基础）](06\. matplotlib/06.03  working with text - basic.ipynb)
    *   [06.04 处理文本（数学表达式）](06\. matplotlib/06.04 working with text - math expression.ipynb)
    *   [06.05 图像基础](06\. matplotlib/06.05 image tutorial.ipynb)
    *   [06.06 注释](06\. matplotlib/06.06 annotating axes.ipynb)
    *   [06.07 标签](06\. matplotlib/06.07 legend.ipynb)
    *   [06.08 figures, subplots, axes 和 ticks 对象](06\. matplotlib/06.08 figures, subplots, axes and ticks.ipynb)
    *   [06.09 不要迷信默认设置](06\. matplotlib/06.09 do not trust the defaults.ipynb)
    *   [06.10 各种绘图实例](06\. matplotlib/06.10 different plots.ipynb)
*   [07\. **使用其他语言进行扩展**](07\. interfacing with other languages)
    *   [07.01 简介](07\. interfacing with other languages/07.01 introduction.ipynb)
    *   [07.02 Python 扩展模块](07\. interfacing with other languages/07.02 python extension modules.ipynb)
    *   [07.03 Cython：Cython 基础，将源代码转换成扩展模块](07\. interfacing with other languages/07.03 cython part 1.ipynb)
    *   [07.04 Cython：Cython 语法，调用其他C库](07\. interfacing with other languages/07.04 cython part 2.ipynb)
    *   [07.05 Cython：class 和 cdef class，使用 C++](07\. interfacing with other languages/07.05 cython part 3.ipynb)
    *   [07.06 Cython：Typed memoryviews](07\. interfacing with other languages/07.06 cython part 4.ipynb)
    *   [07.07 生成编译注释](07\. interfacing with other languages/07.07 profiling with annotations.ipynb)
    *   [07.08 ctypes](07\. interfacing with other languages/07.08 ctypes.ipynb)
*   [08\. **面向对象编程**](08\. object-oriented programming)
    *   [08.01 简介](08\. object-oriented programming/08.01 oop introduction.ipynb)
    *   [08.02 使用 OOP 对森林火灾建模](08\. object-oriented programming/08.02 using oop model a forest fire.ipynb)
    *   [08.03 什么是对象？](08\. object-oriented programming/08.03 what is a object.ipynb)
    *   [08.04 定义 class](08\. object-oriented programming/08.04 writing classes.ipynb)
    *   [08.05 特殊方法](08\. object-oriented programming/08.05 special method.ipynb)
    *   [08.06 属性](08\. object-oriented programming/08.06 properties.ipynb)
    *   [08.07 森林火灾模拟](08\. object-oriented programming/08.07 forest fire simulation.ipynb)
    *   [08.08 继承](08\. object-oriented programming/08.08 inheritance.ipynb)
    *   [08.09 super() 函数](08\. object-oriented programming/08.09 super.ipynb)
    *   [08.10 重定义森林火灾模拟](08\. object-oriented programming/08.10 refactoring the forest fire simutation.ipynb)
    *   [08.11 接口](08\. object-oriented programming/08.11 interfaces.ipynb)
    *   [08.12 共有，私有和特殊方法和属性](08\. object-oriented programming/08.12 public private special in python.ipynb)
    *   [08.13 多重继承](08\. object-oriented programming/08.13 multiple inheritance.ipynb)
*   [09\. **Theano 基础**](09\. theano)
    *   [09.01 Theano 简介及其安装](09\. theano/09.01 introduction and installation.ipynb)
    *   [09.02 Theano 基础](09\. theano/09.02 theano basics.ipynb)
    *   [09.03 Theano 在 Windows 上的配置](09\. theano/09.03 gpu on windows.ipynb)
    *   [09.04 Theano 符号图结构](09\. theano/09.04 graph structures.ipynb)
    *   [09.05 Theano 配置和编译模式](09\. theano/09.05 configuration settings and compiling modes.ipynb)
    *   [09.06 Theano 条件语句](09\. theano/09.06 conditions in theano.ipynb)
    *   [09.07 Theano 循环：scan（详解）](09\. theano/09.07 loop with scan.ipynb)
    *   [09.08 Theano 实例：线性回归](09\. theano/09.08 linear regression.ipynb)
    *   [09.09 Theano 实例：Logistic 回归](09\. theano/09.09 logistic regression .ipynb)
    *   [09.10 Theano 实例：Softmax 回归](09\. theano/09.10 softmax on mnist.ipynb)
    *   [09.11 Theano 实例：人工神经网络](09\. theano/09.11 net on mnist.ipynb)
    *   [09.12 Theano 随机数流变量](09\. theano/09.12 random streams.ipynb)
    *   [09.13 Theano 实例：更复杂的网络](09\. theano/09.13 modern net on mnist.ipynb)
    *   [09.14 Theano 实例：卷积神经网络](09\. theano/09.14 convolutional net on mnist.ipynb)
    *   [09.15 Theano tensor 模块：基础](09\. theano/09.15 tensor basics.ipynb)
    *   [09.16 Theano tensor 模块：索引](09\. theano/09.16 tensor indexing.ipynb)
    *   [09.17 Theano tensor 模块：操作符和逐元素操作](09\. theano/09.17 tensor operator and elementwise operations.ipynb)
    *   [09.18 Theano tensor 模块：nnet 子模块](09\. theano/09.18 tensor nnet .ipynb)
    *   [09.19 Theano tensor 模块：conv 子模块](09\. theano/09.19 tensor conv.ipynb)
*   [10\. **有趣的第三方模块**](10\. something interesting)
    *   [10.01 使用 basemap 画地图](10\. something interesting/10.01 maps using basemap.ipynb)
    *   [10.02 使用 cartopy 画地图](10\. something interesting/10.02 maps using cartopy.ipynb)
    *   [10.03 探索 NBA 数据](10\. something interesting/10.03 nba data.ipynb)
    *   [10.04 金庸的武侠世界](10\. something interesting/10.04 louis cha's kungfu world.ipynb)
*   [11\. **有用的工具**](11\. useful tools)
    *   [11.01 pprint 模块：打印 Python 对象](11\. useful tools/11.01 pprint.ipynb)
    *   [11.02 pickle, cPickle 模块：序列化 Python 对象](11\. useful tools/11.02 pickle and cPickle.ipynb)
    *   [11.03 json 模块：处理 JSON 数据](11\. useful tools/11.03 json.ipynb)
    *   [11.04 glob 模块：文件模式匹配](11\. useful tools/11.04 glob.ipynb)
    *   [11.05 shutil 模块：高级文件操作](11\. useful tools/11.05 shutil.ipynb)
    *   [11.06 gzip, zipfile, tarfile 模块：处理压缩文件](11\. useful tools/11.06 gzip, zipfile, tarfile.ipynb)
    *   [11.07 logging 模块：记录日志](11\. useful tools/11.07 logging.ipynb)
    *   [11.08 string 模块：字符串处理](11\. useful tools/11.08 string.ipynb)
    *   [11.09 collections 模块：更多数据结构](11\. useful tools/11.09 collections.ipynb)
    *   [11.10 requests 模块：HTTP for Human](11\. useful tools/11.10 requests.ipynb)
*   [12\. **Pandas**](12\. pandas)
    *   [12.01 十分钟上手 Pandas](12\. pandas/12.01 ten minutes to pandas.ipynb)
    *   [12.02 一维数据结构：Series](12\. pandas/12.02 series in pandas.ipynb)
    *   [12.03 二维数据结构：DataFrame](12\. pandas/12.03 dataframe in pandas.ipynb)