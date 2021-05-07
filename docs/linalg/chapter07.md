
# 第七讲：求解$Ax=0$，主变量，特解

举例：$3 \times 4$矩阵
$
A=
\begin{bmatrix}
1 & 2 & 2 & 2\\
2 & 4 & 6 & 8\\
3 & 6 & 8 & 10\\
\end{bmatrix}
$，求$Ax=0$的特解：

找出主变量（pivot variable）：
$$
A=
\begin{bmatrix}
1 & 2 & 2 & 2\\
2 & 4 & 6 & 8\\
3 & 6 & 8 & 10\\
\end{bmatrix}
\underrightarrow{消元}
\begin{bmatrix}
\underline{1} & 2 & 2 & 2\\
0 & 0 & \underline{2} & 4\\
0 & 0 & 0 & 0\\
\end{bmatrix}
=U
$$

主变量（pivot variable，下划线元素）的个数为2，即矩阵$A$的秩（rank）为2，即$r=2$。

主变量所在的列为主列（pivot column），其余列为自由列（free column）。

自由列中的变量为自由变量（free variable），自由变量的个数为$n-r=4-2=2$。

通常，给自由列变量赋值，去求主列变量的值。如，令$x_2=1, x_4=0$求得特解
$x=c_1\begin{bmatrix}-2\\1\\0\\0\\\end{bmatrix}$；
再令$x_2=0, x_4=1$求得特解
$x=c_2\begin{bmatrix}2\\0\\-2\\1\\\end{bmatrix}$。

该例还能进一步简化，即将$U$矩阵化简为$R$矩阵（Reduced row echelon form），即简化行阶梯形式。

在简化行阶梯形式中，主元上下的元素都是$0$：
$$
U=
\begin{bmatrix}
\underline{1} & 2 & 2 & 2\\
0 & 0 & \underline{2} & 4\\
0 & 0 & 0 & 0\\
\end{bmatrix}
\underrightarrow{化简}
\begin{bmatrix}
\underline{1} & 2 & 0 & -2\\
0 & 0 & \underline{1} & 2\\
0 & 0 & 0 & 0\\
\end{bmatrix}
=R
$$

将$R$矩阵中的主变量放在一起，自由变量放在一起（列交换），得到

$$
R=
\begin{bmatrix}
\underline{1} & 2 & 0 & -2\\
0 & 0 & \underline{1} & 2\\
0 & 0 & 0 & 0\\
\end{bmatrix}
\underrightarrow{列交换}
\left[
\begin{array}{c c | c c}
1 & 0 & 2 & -2\\
0 & 1 & 0 & 2\\
\hline
0 & 0 & 0 & 0\\
\end{array}
\right]
=
\begin{bmatrix}
I & F \\
0 & 0 \\
\end{bmatrix}
\textrm{，其中}I\textrm{为单位矩阵，}F\textrm{为自由变量组成的矩阵}
$$

计算零空间矩阵$N$（nullspace matrix），其列为特解，有$RN=0$。

$$
x_{pivot}=-Fx_{free} \\
\begin{bmatrix}
I & F \\
\end{bmatrix}
\begin{bmatrix}
x_{pivot} \\
x_{free} \\
\end{bmatrix}=0 \\
N=\begin{bmatrix}
-F \\
I \\
\end{bmatrix}
$$

在本例中
$
N=
\begin{bmatrix}
-2 & 2 \\
0 & -2 \\
1 & 0 \\
0 & 1 \\
\end{bmatrix}
$，与上面求得的两个$x$特解一致。

另一个例子，矩阵
$
A=
\begin{bmatrix}
1 & 2 & 3 \\
2 & 4 & 6 \\
2 & 6 & 8 \\
2 & 8 & 10 \\
\end{bmatrix}
\underrightarrow{消元}
\begin{bmatrix}
1 & 2 & 3 \\
0 & 2 & 2 \\
0 & 0 & 0 \\
0 & 0 & 0 \\
\end{bmatrix}
\underrightarrow{化简}
\begin{bmatrix}
1 & 0 & 1 \\
0 & 1 & 1 \\
0 & 0 & 0 \\
0 & 0 & 0 \\
\end{bmatrix}
=R
$

矩阵的秩仍为$r=2$，有$2$个主变量，$1$个自由变量。

同上一例，取自由变量为$x_3=1$，求得特解
$
x=c
\begin{bmatrix}
-1 \\
-1 \\
1 \\
\end{bmatrix}
$
