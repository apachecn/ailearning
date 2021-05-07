
# 第二十六讲：对称矩阵及正定性

## 对称矩阵

前面我们学习了矩阵的特征值与特征向量，也了解了一些特殊的矩阵及其特征值、特征向量，特殊矩阵的特殊性应该会反映在其特征值、特征向量中。如马尔科夫矩阵，有一特征值为$1$，本讲介绍（实）对称矩阵。

先提前介绍两个对称矩阵的特性：

1. 特征值为实数；（对比第二十一讲介绍的旋转矩阵，其特征值为纯虚数。）
2. 特征向量相互正交。（当特征值重复时，特征向量也可以从子空间中选出相互正交正交的向量。）

典型的状况是，特征值不重复，特征向量相互正交。

* 那么在通常（可对角化）情况下，一个矩阵可以化为：$A=S\varLambda S^{-1}$；
* 在矩阵对称的情况下，通过性质2可知，由特征向量组成的矩阵$S$中的列向量是相互正交的，此时如果我们把特征向量的长度统一化为$1$，就可以得到一组标准正交的特征向量。则对于对称矩阵有$A=Q\varLambda Q^{-1}$，而对于标准正交矩阵，有$Q=Q^T$，所以对称矩阵可以写为$$A=Q\varLambda Q^T\tag{1}$$

观察$(1)$式，我们发现这个分解本身就代表着对称，$\left(Q\varLambda Q^T\right)^T=\left(Q^T\right)^T\varLambda^TQ^T=Q\varLambda Q^T$。$(1)$式在数学上叫做谱定理（spectral theorem），谱就是指矩阵特征值的集合。（该名称来自光谱，指一些纯事物的集合，就像将特征值分解成为特征值与特征向量。）在力学上称之为主轴定理（principle axis theorem），从几何上看，它意味着如果给定某种材料，在合适的轴上来看，它就变成对角化的，方向就不会重复。

* 现在我们来证明性质1。对于矩阵$\underline{Ax=\lambda x}$，对于其共轭部分总有$\bar A\bar x=\bar\lambda \bar x$，根据前提条件我们只讨论实矩阵，则有$A\bar x=\bar\lambda \bar x$，将等式两边取转置有$\overline{\bar{x}^TA=\bar{x}^T\bar\lambda}$。将“下划线”式两边左乘$\bar{x}^T$有$\bar{x}^TAx=\bar{x}^T\lambda x$，“上划线”式两边右乘$x$有$\bar{x}^TAx=\bar{x}^T\bar\lambda x$，观察发现这两个式子左边是一样的，所以$\bar{x}^T\lambda x=\bar{x}^T\bar\lambda x$，则有$\lambda=\bar{\lambda}$（这里有个条件，$\bar{x}^Tx\neq 0$），证毕。

    观察这个前提条件，$\bar{x}^Tx=\begin{bmatrix}\bar x_1&\bar x_2&\cdots&\bar x_n\end{bmatrix}\begin{bmatrix}x_1\\x_2\\\vdots\\x_n\end{bmatrix}=\bar x_1x_1+\bar x_2x_2+\cdots+\bar x_nx_n$，设$x_1=a+ib, \bar x_1=a-ib$则$\bar x_1x_1=a^2+b^2$，所以有$\bar{x}^Tx>0$。而$\bar{x}^Tx$就是$x$长度的平方。

    拓展这个性质，当$A$为复矩阵，根据上面的推导，则矩阵必须满足$A=\bar{A}^T$时，才有性质1、性质2成立（教授称具有这种特征值为实数、特征向量相互正交的矩阵为“好矩阵”）。

继续研究$A=Q\varLambda Q^T=\Bigg[q_1\ q_2\ \cdots\ q_n\Bigg]\begin{bmatrix}\lambda_1& &\cdots& \\&\lambda_2&\cdots&\\\vdots&\vdots&\ddots&\vdots\\& &\cdots&\lambda_n\end{bmatrix}\begin{bmatrix}\quad q_1^T\quad\\\quad q_1^T\quad\\\quad \vdots \quad\\\quad q_1^T\quad\end{bmatrix}=\lambda_1q_1q_1^T+\lambda_2q_2q_2^T+\cdots+\lambda_nq_nq_n^T$，注意这个展开式中的$qq^T$，$q$是单位列向量所以$q^Tq=1$，结合我们在第十五讲所学的投影矩阵的知识有$\frac{qq^T}{q^Tq}=qq^T$是一个投影矩阵，很容易验证其性质，比如平方它会得到$qq^Tqq^T=qq^T$于是多次投影不变等。

**每一个对称矩阵都可以分解为一系列相互正交的投影矩阵。**

在知道对称矩阵的特征值皆为实数后，我们再来讨论这些实数的符号，因为特征值的正负号会影响微分方程的收敛情况（第二十三讲，需要实部为负的特征值保证收敛）。用消元法取得矩阵的主元，观察主元的符号，**主元符号的正负数量与特征向量的正负数量相同**。

## 正定性

如果对称矩阵是“好矩阵”，则正定矩阵（positive definite）是其一个更好的子类。正定矩阵指特征值均为正数的矩阵（根据上面的性质有矩阵的主元均为正）。

举个例子，$\begin{bmatrix}5&2\\2&3\end{bmatrix}$，由行列式消元知其主元为$5,\frac{11}{5}$，按一般的方法求特征值有$\begin{vmatrix}5-\lambda&2\\2&3-lambda\end{vmatrix}=\lambda^2-8\lambda+11=0, \lambda=4\pm\sqrt 5$。

正定矩阵的另一个性质是，所有子行列式为正。对上面的例子有$\begin{vmatrix}5\end{vmatrix}=5, \begin{vmatrix}5&2\\2&3\end{vmatrix}=11$。

我们看到正定矩阵将早期学习的的消元主元、中期学习的的行列式、后期学习的特征值结合在了一起。
