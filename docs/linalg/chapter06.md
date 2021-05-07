
# 第六讲：列空间和零空间

对向量子空间$S$和$T$，有$S \cap T$也是向量子空间。

对$m \times n$矩阵$A$，$n \times 1$矩阵$x$，$m \times 1$矩阵$b$，运算$Ax=b$：

$$
\begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1(n-1)} & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2(n-1)} & a_{2n} \\
\vdots & \vdots & \ddots & \vdots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{m(n-1)} & a_{mn} \\
\end{bmatrix}
\cdot
\begin{bmatrix}
x_{1} \\
x_{2} \\
\vdots \\
x_{n-1} \\
x_{n} \\
\end{bmatrix}
=
\begin{bmatrix}
b_{1} \\
b_{2} \\
\vdots \\
b_{m} \\
\end{bmatrix}
$$

由$A$的列向量生成的子空间为$A$的列空间；

$Ax=b$有非零解当且仅当$b$属于$A$的列空间

A的零空间是$Ax=0$中$x$的解组成的集合。
