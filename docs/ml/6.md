# 第6章 支持向量机
<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

![支持向量机_首页](img/SVM_1.jpg)

## 支持向量机 概述

支持向量机(Support Vector Machines, SVM): 是一种监督学习算法。
* 支持向量(Support Vector)就是离分隔超平面最近的那些点。
* 机(Machine)就是表示一种算法，而不是表示机器。

## 支持向量机 场景

* 要给左右两边的点进行分类
* 明显发现: 选择D会比B、C分隔的效果要好很多。

![线性可分](img/SVM_3_linearly-separable.jpg)


## 支持向量机 原理

### SVM 工作原理

![k_2](img/k_2.jpg "k_2")

对于上述的苹果和香蕉，我们想象为2种水果类型的炸弹。（保证距离最近的炸弹，距离它们最远）

1. 寻找最大分类间距
2. 转而通过拉格朗日函数求优化的问题


* 数据可以通过画一条直线就可以将它们完全分开，这组数据叫`线性可分(linearly separable)`数据，而这条分隔直线称为`分隔超平面(separating hyperplane)`。
* 如果数据集上升到1024维呢？那么需要1023维来分隔数据集，也就说需要N-1维的对象来分隔，这个对象叫做`超平面(hyperlane)`，也就是分类的决策边界。

![分隔超平面](img/SVM_2_separating-hyperplane.jpg)

### 寻找最大间隔

#### 为什么寻找最大间隔

```
摘录地址: http://slideplayer.com/slide/8610144  (第12条信息)
Support Vector Machines: Slide 12 Copyright © 2001, 2003, Andrew W. Moore Why Maximum Margin? 

1.Intuitively this feels safest. 
2.If we’ve made a small error in the location of the boundary (it’s been jolted in its perpendicular direction) this gives us least chance of causing a misclassification. 
3.CV is easy since the model is immune to removal of any non-support-vector datapoints. 
4.There’s some theory that this is a good thing. 
5.Empirically it works very very well. 

* * *

1. 直觉上是最安全的
2. 如果我们在边界的位置发生了一个小错误（它在垂直方向上被颠倒），这给我们最小的可能导致错误分类。
3. CV（cross validation 交叉验证）很容易，因为该模型对任何非支持向量数据点的去除是免疫的。
4. 有一些理论表明这是一件好东西。
5. 从经验角度上说它的效果非常非常好。
```

#### 怎么寻找最大间隔

> 点到超平面的距离

* 分隔超平面`函数间距`:  $$y(x)=w^Tx+b$$
* 分类的结果:  $$f(x)=sign(w^Tx+b)$$  (sign表示>0为1，<0为-1，=0为0) 
* 点到超平面的`几何间距`: $$d(x)=(w^Tx+b)/||w||$$  （||w||表示w矩阵的二范数=> $$\sqrt{w^T*w}$$, 点到超平面的距离也是类似的）

![点到直线的几何距离](img/SVM_4_point2line-distance.jpg)

> 拉格朗日乘子法

* 类别标签用-1、1，是为了后期方便 $$label*(w^Tx+b)$$ 的标识和距离计算；如果 $$label*(w^Tx+b)>0$$ 表示预测正确，否则预测错误。
* 现在目标很明确，就是要找到`w`和`b`，因此我们必须要找到最小间隔的数据点，也就是前面所说的`支持向量`。
    * 也就说，让最小的距离取最大.(最小的距离: 就是最小间隔的数据点；最大: 就是最大间距，为了找出最优超平面--最终就是支持向量)
    * 目标函数: $$arg: max_{w, b} \left( min[label*(w^Tx+b)]*\frac{1}{||w||} \right) $$
        1. 如果 $$label*(w^Tx+b)>0$$ 表示预测正确，也称`函数间隔`，$$||w||$$ 可以理解为归一化，也称`几何间隔`。
        2. 令 $$label*(w^Tx+b)>=1$$， 因为0～1之间，得到的点是存在误判的可能性，所以要保障 $$min[label*(w^Tx+b)]=1$$，才能更好降低噪音数据影响。
        3. 所以本质上是求 $$arg: max_{w, b}  \frac{1}{||w||} $$；也就说，我们约束(前提)条件是: $$label*(w^Tx+b)=1$$
* 新的目标函数求解:  $$arg: max_{w, b}  \frac{1}{||w||} $$
    * => 就是求: $$arg: min_{w, b} ||w|| $$ (求矩阵会比较麻烦，如果x只是 $$\frac{1}{2}*x^2$$ 的偏导数，那么。。同样是求最小值)
    * => 就是求: $$arg: min_{w, b} (\frac{1}{2}*||w||^2)$$ (二次函数求导，求极值，平方也方便计算)
    * 本质上就是求线性不等式的二次优化问题(求分隔超平面，等价于求解相应的凸二次规划问题)
* 通过拉格朗日乘子法，求二次优化问题
    * 假设需要求极值的目标函数 (objective function) 为 f(x,y)，限制条件为 φ(x,y)=M  # M=1
    * 设g(x,y)=M-φ(x,y)   # 临时φ(x,y)表示下文中 $$label*(w^Tx+b)$$
    * 定义一个新函数: F(x,y,λ)=f(x,y)+λg(x,y)
    * a为λ（a>=0），代表要引入的拉格朗日乘子(Lagrange multiplier)
    * 那么:  $$L(w,b,\alpha)=\frac{1}{2} * ||w||^2 + \sum_{i=1}^{n} \alpha_i * [1 - label * (w^Tx+b)]$$
    * 因为: $$label*(w^Tx+b)>=1, \alpha>=0$$ , 所以 $$\alpha*[1-label*(w^Tx+b)]<=0$$ , $$\sum_{i=1}^{n} \alpha_i * [1-label*(w^Tx+b)]<=0$$ 
    * 当 $$label*(w^Tx+b)>1$$ 则 $$\alpha=0$$ ，表示该点为<font color=red>非支持向量</font>
    * 相当于求解:  $$max_{\alpha} L(w,b,\alpha) = \frac{1}{2} *||w||^2$$ 
    * 如果求:  $$min_{w, b} \frac{1}{2} *||w||^2$$ , 也就是要求:  $$min_{w, b} \left( max_{\alpha} L(w,b,\alpha)\right)$$ 
* 现在转化到对偶问题的求解
    * $$min_{w, b} \left(max_{\alpha} L(w,b,\alpha) \right) $$ >= $$max_{\alpha} \left(min_{w, b}\ L(w,b,\alpha) \right) $$ 
    * 现在分2步
    * 先求:  $$min_{w, b} L(w,b,\alpha)=\frac{1}{2} * ||w||^2 + \sum_{i=1}^{n} \alpha_i * [1 - label * (w^Tx+b)]$$
    * 就是求`L(w,b,a)`关于[w, b]的偏导数, 得到`w和b的值`，并化简为: `L和a的方程`。
    * 参考:  如果公式推导还是不懂，也可以参考《统计学习方法》李航-P103<学习的对偶算法>
    ![计算拉格朗日函数的对偶函数](img/SVM_5_Lagrangemultiplier.png)
* 终于得到课本上的公式:  $$max_{\alpha} \left( \sum_{i=1}^{m} \alpha_i - \frac{1}{2} \sum_{i, j=1}^{m} label_i \ast label_j \ast \alpha_i \ast \alpha_j \ast <x_i, x_j> \right) $$
* 约束条件:  $$a>=0$$ 并且 $$\sum_{i=1}^{m} a_i \ast label_i=0$$

> 松弛变量(slack variable)

参考地址: http://blog.csdn.net/wusecaiyun/article/details/49659183

![松弛变量公式](img/SVM_松弛变量.jpg)

* 我们知道几乎所有的数据都不那么干净, 通过引入松弛变量来 `允许数据点可以处于分隔面错误的一侧`。
* 约束条件:  $$C>=a>=0$$ 并且 $$\sum_{i=1}^{m} a_i \ast label_i=0$$
* 总的来说: 
    * ![松弛变量](img/松弛变量.png) 表示 `松弛变量`
    * 常量C是 `惩罚因子`, 表示离群点的权重（用于控制“最大化间隔”和“保证大部分点的函数间隔小于1.0” ）
        * $$label*(w^Tx+b) > 1$$ and alpha = 0 (在边界外，就是非支持向量)
        * $$label*(w^Tx+b) = 1$$ and 0< alpha < C (在分割超平面上，就支持向量)
        * $$label*(w^Tx+b) < 1$$ and alpha = C (在分割超平面内，是误差点 -> C表示它该受到的惩罚因子程度)
        * 参考地址: https://www.zhihu.com/question/48351234/answer/110486455
    * C值越大，表示离群点影响越大，就越容易过度拟合；反之有可能欠拟合。
    * 我们看到，目标函数控制了离群点的数目和程度，使大部分样本点仍然遵守限制条件。
    * 例如: 正类有10000个样本，而负类只给了100个（C越大表示100个负样本的影响越大，就会出现过度拟合，所以C决定了负样本对模型拟合程度的影响！，C就是一个非常关键的优化点！）
* 这一结论十分直接，SVM中的主要工作就是要求解 alpha.

### SMO 高效优化算法

* SVM有很多种实现，最流行的一种实现是:  `序列最小优化(Sequential Minimal Optimization, SMO)算法`。
* 下面还会介绍一种称为 `核函数(kernel)` 的方式将SVM扩展到更多数据集上。
* 注意: `SVM几何含义比较直观，但其算法实现较复杂，牵扯大量数学公式的推导。`

> 序列最小优化(Sequential Minimal Optimization, SMO)

* 创建作者: John Platt
* 创建时间: 1996年
* SMO用途: 用于训练 SVM
* SMO目标: 求出一系列 alpha 和 b,一旦求出 alpha，就很容易计算出权重向量 w 并得到分隔超平面。
* SMO思想: 是将大优化问题分解为多个小优化问题来求解的。
* SMO原理: 每次循环选择两个 alpha 进行优化处理，一旦找出一对合适的 alpha，那么就增大一个同时减少一个。
    * 这里指的合适必须要符合一定的条件
        1. 这两个 alpha 必须要在间隔边界之外
        2. 这两个 alpha 还没有进行过区间化处理或者不在边界上。
    * 之所以要同时改变2个 alpha；原因是我们有一个约束条件:  $$\sum_{i=1}^{m} a_i \ast label_i=0$$；如果只是修改一个 alpha，很可能导致约束条件失效。

> SMO 伪代码大致如下: 

```
创建一个 alpha 向量并将其初始化为0向量
当迭代次数小于最大迭代次数时(外循环)
    对数据集中的每个数据向量(内循环): 
        如果该数据向量可以被优化
            随机选择另外一个数据向量
            同时优化这两个向量
            如果两个向量都不能被优化，退出内循环
    如果所有向量都没被优化，增加迭代数目，继续下一次循环
```

### SVM 开发流程

```
收集数据: 可以使用任意方法。
准备数据: 需要数值型数据。
分析数据: 有助于可视化分隔超平面。
训练算法: SVM的大部分时间都源自训练，该过程主要实现两个参数的调优。
测试算法: 十分简单的计算过程就可以实现。
使用算法: 几乎所有分类问题都可以使用SVM，值得一提的是，SVM本身是一个二类分类器，对多类问题应用SVM需要对代码做一些修改。
```

### SVM 算法特点

```
优点: 泛化（由具体的、个别的扩大为一般的，就是说: 模型训练完后的新样本）错误率低，计算开销不大，结果易理解。
缺点: 对参数调节和核函数的选择敏感，原始分类器不加修改仅适合于处理二分类问题。
使用数据类型: 数值型和标称型数据。
```

### 课本案例（无核函数）

#### 项目概述

对小规模数据点进行分类

#### 开发流程

> 收集数据

文本文件格式: 

```python
3.542485	1.977398	-1
3.018896	2.556416	-1
7.551510	-1.580030	1
2.114999	-0.004466	-1
8.127113	1.274372	1
```

> 准备数据

```python
def loadDataSet(fileName):
    """
    对文件进行逐行解析，从而得到第行的类标签和整个特征矩阵
    Args:
        fileName 文件名
    Returns:
        dataMat  特征矩阵
        labelMat 类标签
    """
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat
```

> 分析数据: 无

> 训练算法

```python
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    """smoSimple

    Args:
        dataMatIn    特征集合
        classLabels  类别标签
        C   松弛变量(常量值)，允许有些数据点可以处于分隔面的错误一侧。
            控制最大化间隔和保证大部分的函数间隔小于1.0这两个目标的权重。
            可以通过调节该参数达到不同的结果。
        toler   容错率（是指在某个体系中能减小一些因素或选择对某个系统产生不稳定的概率。）
        maxIter 退出前最大的循环次数
    Returns:
        b       模型的常量值
        alphas  拉格朗日乘子
    """
    dataMatrix = mat(dataMatIn)
    # 矩阵转置 和 .T 一样的功能
    labelMat = mat(classLabels).transpose()
    m, n = shape(dataMatrix)

    # 初始化 b和alphas(alpha有点类似权重值。)
    b = 0
    alphas = mat(zeros((m, 1)))

    # 没有任何alpha改变的情况下遍历数据的次数
    iter = 0
    while (iter < maxIter):
        # w = calcWs(alphas, dataMatIn, classLabels)
        # print("w:", w)

        # 记录alpha是否已经进行优化，每次循环时设为0，然后再对整个集合顺序遍历
        alphaPairsChanged = 0
        for i in range(m):
            # print 'alphas=', alphas
            # print 'labelMat=', labelMat
            # print 'multiply(alphas, labelMat)=', multiply(alphas, labelMat)
            # 我们预测的类别 y[i] = w^Tx[i]+b; 其中因为 w = Σ(1~n) a[n]*label[n]*x[n]
            fXi = float(multiply(alphas, labelMat).T*(dataMatrix*dataMatrix[i, :].T)) + b
            # 预测结果与真实结果比对，计算误差Ei
            Ei = fXi - float(labelMat[i])

            # 约束条件 (KKT条件是解决最优化问题的时用到的一种方法。我们这里提到的最优化问题通常是指对于给定的某一函数，求其在指定作用域上的全局最小值)
            # 0<=alphas[i]<=C，但由于0和C是边界值，我们无法进行优化，因为需要增加一个alphas和降低一个alphas。
            # 表示发生错误的概率: labelMat[i]*Ei 如果超出了 toler， 才需要优化。至于正负号，我们考虑绝对值就对了。
            '''
            # 检验训练样本(xi, yi)是否满足KKT条件
            yi*f(i) >= 1 and alpha = 0 (outside the boundary)
            yi*f(i) == 1 and 0<alpha< C (on the boundary)
            yi*f(i) <= 1 and alpha = C (between the boundary)
            '''
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):

                # 如果满足优化的条件，我们就随机选取非i的一个点，进行优化比较
                j = selectJrand(i, m)
                # 预测j的结果
                fXj = float(multiply(alphas, labelMat).T*(dataMatrix*dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()

                # L和H用于将alphas[j]调整到0-C之间。如果L==H，就不做任何改变，直接执行continue语句
                # labelMat[i] != labelMat[j] 表示异侧，就相减，否则是同侧，就相加。
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                # 如果相同，就没法优化了
                if L == H:
                    print("L==H")
                    continue

                # eta是alphas[j]的最优修改量，如果eta==0，需要退出for循环的当前迭代过程
                # 参考《统计学习方法》李航-P125~P128<序列最小最优化算法>
                eta = 2.0 * dataMatrix[i, :]*dataMatrix[j, :].T - dataMatrix[i, :]*dataMatrix[i, :].T - dataMatrix[j, :]*dataMatrix[j, :].T
                if eta >= 0:
                    print("eta>=0")
                    continue

                # 计算出一个新的alphas[j]值
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                # 并使用辅助函数，以及L和H对其进行调整
                alphas[j] = clipAlpha(alphas[j], H, L)
                # 检查alpha[j]是否只是轻微的改变，如果是的话，就退出for循环。
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print("j not moving enough")
                    continue
                # 然后alphas[i]和alphas[j]同样进行改变，虽然改变的大小一样，但是改变的方向正好相反
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])
                # 在对alpha[i], alpha[j] 进行优化之后，给这两个alpha值设置一个常数b。
                # w= Σ[1~n] ai*yi*xi => b = yj- Σ[1~n] ai*yi(xi*xj)
                # 所以:   b1 - b = (y1-y) - Σ[1~n] yi*(a1-a)*(xi*x1)
                # 为什么减2遍？ 因为是 减去Σ[1~n]，正好2个变量i和j，所以减2遍
                b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i, :]*dataMatrix[i, :].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i, :]*dataMatrix[j, :].T
                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i, :]*dataMatrix[j, :].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j, :]*dataMatrix[j, :].T
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2)/2.0
                alphaPairsChanged += 1
                print("iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
        # 在for循环外，检查alpha值是否做了更新，如果更新则将iter设为0后继续运行程序
        # 直到更新完毕后，iter次循环无变化，才退出循环。
        if (alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
        print("iteration number: %d" % iter)
    return b, alphas
```

[完整代码地址: SVM简化版，应用简化版SMO算法处理小规模数据集](https://github.com/apachecn/AiLearning/blob/master/src/py2.x/ml/6.SVM/svm-simple.py): <https://github.com/apachecn/AiLearning/blob/master/src/py2.x/ml/6.SVM/svm-simple.py>

[完整代码地址: SVM完整版，使用完整 Platt SMO算法加速优化，优化点: 选择alpha的方式不同](https://github.com/apachecn/AiLearning/blob/master/src/py2.x/ml/6.SVM/svm-complete_Non-Kernel.py): <https://github.com/apachecn/AiLearning/blob/master/src/py2.x/ml/6.SVM/svm-complete_Non-Kernel.py>

## 核函数(kernel) 使用

* 对于线性可分的情况，效果明显
* 对于非线性的情况也一样，此时需要用到一种叫`核函数(kernel)`的工具将数据转化为分类器易于理解的形式。

> 利用核函数将数据映射到高维空间

* 使用核函数: 可以将数据从某个特征空间到另一个特征空间的映射。（通常情况下: 这种映射会将低维特征空间映射到高维空间。）
* 如果觉得特征空间很装逼、很难理解。
* 可以把核函数想象成一个包装器(wrapper)或者是接口(interface)，它能将数据从某个很难处理的形式转换成为另一个较容易处理的形式。
* 经过空间转换后: 低维需要解决的非线性问题，就变成了高维需要解决的线性问题。
* SVM 优化特别好的地方，在于所有的运算都可以写成内积(inner product: 是指2个向量相乘，得到单个标量 或者 数值)；内积替换成核函数的方式被称为`核技巧(kernel trick)`或者`核"变电"(kernel substation)`
* 核函数并不仅仅应用于支持向量机，很多其他的机器学习算法也都用到核函数。最流行的核函数: 径向基函数(radial basis function)
* 径向基函数的高斯版本，其具体的公式为: 

![径向基函数的高斯版本](img/SVM_6_radial-basis-function.jpg)

### 项目案例: 手写数字识别的优化（有核函数）

#### 项目概述

```python
你的老板要求: 你写的那个手写识别程序非常好，但是它占用内存太大。顾客无法通过无线的方式下载我们的应用。
所以: 我们可以考虑使用支持向量机，保留支持向量就行（knn需要保留所有的向量），就可以获得非常好的效果。
```

#### 开发流程

> 收集数据: 提供的文本文件

```python
00000000000000001111000000000000
00000000000000011111111000000000
00000000000000011111111100000000
00000000000000011111111110000000
00000000000000011111111110000000
00000000000000111111111100000000
00000000000000111111111100000000
00000000000001111111111100000000
00000000000000111111111100000000
00000000000000111111111100000000
00000000000000111111111000000000
00000000000001111111111000000000
00000000000011111111111000000000
00000000000111111111110000000000
00000000001111111111111000000000
00000001111111111111111000000000
00000011111111111111110000000000
00000111111111111111110000000000
00000111111111111111110000000000
00000001111111111111110000000000
00000001111111011111110000000000
00000000111100011111110000000000
00000000000000011111110000000000
00000000000000011111100000000000
00000000000000111111110000000000
00000000000000011111110000000000
00000000000000011111110000000000
00000000000000011111111000000000
00000000000000011111111000000000
00000000000000011111111000000000
00000000000000000111111110000000
00000000000000000111111100000000

```

> 准备数据: 基于二值图像构造向量

`将 32*32的文本转化为 1*1024的矩阵`

```python
def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect

def loadImages(dirName):
    from os import listdir
    hwLabels = []
    print(dirName)
    trainingFileList = listdir(dirName)  # load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9:
            hwLabels.append(-1)
        else:
            hwLabels.append(1)
        trainingMat[i, :] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels
```

> 分析数据: 对图像向量进行目测

> 训练算法: 采用两种不同的核函数，并对径向基核函数采用不同的设置来运行SMO算法

```python
def kernelTrans(X, A, kTup):  # calc the kernel or transform data to a higher dimensional space
    """
    核转换函数
    Args:
        X     dataMatIn数据集
        A     dataMatIn数据集的第i行的数据
        kTup  核函数的信息

    Returns:

    """
    m, n = shape(X)
    K = mat(zeros((m, 1)))
    if kTup[0] == 'lin':
        # linear kernel:   m*n * n*1 = m*1
        K = X * A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        # 径向基函数的高斯版本
        K = exp(K / (-1 * kTup[1] ** 2))  # divide in NumPy is element-wise not matrix like Matlab
    else:
        raise NameError('Houston We Have a Problem -- That Kernel is not recognized')
    return K

def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    """
    完整SMO算法外循环，与smoSimple有些类似，但这里的循环退出条件更多一些
    Args:
        dataMatIn    数据集
        classLabels  类别标签
        C   松弛变量(常量值)，允许有些数据点可以处于分隔面的错误一侧。
            控制最大化间隔和保证大部分的函数间隔小于1.0这两个目标的权重。
            可以通过调节该参数达到不同的结果。
        toler   容错率
        maxIter 退出前最大的循环次数
        kTup    包含核函数信息的元组
    Returns:
        b       模型的常量值
        alphas  拉格朗日乘子
    """

    # 创建一个 optStruct 对象
    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler, kTup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0

    # 循环遍历: 循环maxIter次 并且 （alphaPairsChanged存在可以改变 or 所有行遍历一遍）
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0

        #  当entireSet=true or 非边界alpha对没有了；就开始寻找 alpha对，然后决定是否要进行else。
        if entireSet:
            # 在数据集上遍历所有可能的alpha
            for i in range(oS.m):
                # 是否存在alpha对，存在就+1
                alphaPairsChanged += innerL(i, oS)
                # print("fullSet, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1

        # 对已存在 alpha对，选出非边界的alpha值，进行优化。
        else:
            # 遍历所有的非边界alpha值，也就是不在边界0或C上的值。
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                # print("non-bound, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1

        # 如果找到alpha对，就优化非边界alpha值，否则，就重新进行寻找，如果寻找一遍 遍历所有的行还是没找到，就退出循环。
        if entireSet:
            entireSet = False  # toggle entire set loop
        elif (alphaPairsChanged == 0):
            entireSet = True
        print("iteration number: %d" % iter)
    return oS.b, oS.alphas
```

> 测试算法: 便携一个函数来测试不同的和函数并计算错误率

```python
def testDigits(kTup=('rbf', 10)):

    # 1. 导入训练数据
    dataArr, labelArr = loadImages('data/6.SVM/trainingDigits')
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    datMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A > 0)[0]
    sVs = datMat[svInd]
    labelSV = labelMat[svInd]
    # print("there are %d Support Vectors" % shape(sVs)[0])
    m, n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], kTup)
        # 1*m * m*1 = 1*1 单个预测结果
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]): errorCount += 1
    print("the training error rate is: %f" % (float(errorCount) / m))

    # 2. 导入测试数据
    dataArr, labelArr = loadImages('data/6.SVM/testDigits')
    errorCount = 0
    datMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    m, n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], kTup)
        # 1*m * m*1 = 1*1 单个预测结果
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]): errorCount += 1
    print("the test error rate is: %f" % (float(errorCount) / m))
```

> 使用算法: 一个图像识别的完整应用还需要一些图像处理的知识，这里并不打算深入介绍

[完整代码地址](https://github.com/apachecn/AiLearning/blob/master/src/py2.x/ml/6.SVM/svm-complete.py): <https://github.com/apachecn/AiLearning/blob/master/src/py2.x/ml/6.SVM/svm-complete.py>

* * *

* **作者: [片刻](http://cwiki.apachecn.org/display/~jiangzhonglian) [geekidentity](http://cwiki.apachecn.org/display/~houfachao)**
* [GitHub地址](https://github.com/apachecn/AiLearning): <https://github.com/apachecn/AiLearning>
* **版权声明: 欢迎转载学习 => 请标注信息来源于 [ApacheCN](http://www.apachecn.org/)**
