
# 第十讲 四个基本子空间

对于$m \times n$矩阵$A$，$rank(A)=r$有：

* 行空间$C(A^T) \in \mathbb{R}^n, dim C(A^T)=r$，基见例1。

* 零空间$N(A) \in \mathbb{R}^n, dim N(A)=n-r$，自由元所在的列即可组成零空间的一组基。

* 列空间$C(A) \in \mathbb{R}^m, dim C(A)=r$，主元所在的列即可组成列空间的一组基。

* 左零空间$N(A^T) \in \mathbb{R}^m, dim N(A^T)=m-r$，基见例2。

例1，对于行空间
$
A=
\begin{bmatrix}
1 & 2 & 3 & 1 \\
1 & 1 & 2 & 1 \\
1 & 2 & 3 & 1 \\
\end{bmatrix}
\underrightarrow{消元、化简}
\begin{bmatrix}
1 & 0 & 1 & 1 \\
0 & 1 & 1 & 0 \\
0 & 0 & 0 & 0 \\
\end{bmatrix}
=R
$

由于我们做了行变换，所以A的列空间受到影响，$C(R) \neq C(A)$，而行变换并不影响行空间，所以可以在$R$中看出前两行就是行空间的一组基。

所以，可以得出无论对于矩阵$A$还是$R$，其行空间的一组基，可以由$R$矩阵的前$r$行向量组成（这里的$R$就是第七讲提到的简化行阶梯形式）。

例2，对于左零空间，有$A^Ty=0 \rightarrow (A^Ty)^T=0^T\rightarrow y^TA=0^T$，因此得名。

采用Gauss-Jordan消元，将增广矩阵$\left[\begin{array}{c|c}A_{m \times n} & I_{m \times m}\end{array}\right]$中$A$的部分划为简化行阶梯形式$\left[\begin{array}{c|c}R_{m \times n} & E_{m \times m}\end{array}\right]$，此时矩阵$E$会将所有的行变换记录下来。

则$EA=R$，而在前几讲中，有当$A'$是$m$阶可逆方阵时，$R'$即是$I$，所以$E$就是$A^{-1}$。

本例中

$$
\left[\begin{array}{c|c}A_{m \times n} & I_{m \times m}\end{array}\right]=
\left[
\begin{array}
{c c c c|c c c}
1 & 2 & 3 & 1 & 1 & 0 & 0 \\
1 & 1 & 2 & 1 & 0 & 1 & 0 \\
1 & 2 & 3 & 1 & 0 & 0 & 1 \\
\end{array}
\right]
\underrightarrow{消元、化简}
\left[
\begin{array}
{c c c c|c c c}
1 & 0 & 1 & 1 & -1 & 2 & 0 \\
0 & 1 & 1 & 0 & 1 & -1 & 0 \\
0 & 0 & 0 & 0 & -1 & 0 & 1 \\
\end{array}
\right]
=\left[\begin{array}{c|c}R_{m \times n} & E_{m \times m}\end{array}\right]
$$

则

$$
EA=
\begin{bmatrix}
-1 & 2  & 0 \\
1  & -1 & 0 \\
-1 & 0  & 1 \\
\end{bmatrix}
\cdot
\begin{bmatrix}
1 & 2 & 3 & 1 \\
1 & 1 & 2 & 1 \\
1 & 2 & 3 & 1 \\
\end{bmatrix}
=
\begin{bmatrix}
1 & 0 & 1 & 1 \\
0 & 1 & 1 & 0 \\
0 & 0 & 0 & 0 \\
\end{bmatrix}
=R
$$


很明显，式中$E$的最后一行对$A$的行做线性组合后，得到$R$的最后一行，即$0$向量，也就是$y^TA=0^T$。

最后，引入矩阵空间的概念，矩阵可以同向量一样，做求和、数乘。

举例，设所有$3 \times 3$矩阵组成的矩阵空间为$M$。则上三角矩阵、对称矩阵、对角矩阵（前两者的交集）。

观察一下对角矩阵，如果取
$
\begin{bmatrix}
1 & 0 & 0 \\
0 & 0 & 0 \\
0 & 0 & 0 \\
\end{bmatrix} \quad
\begin{bmatrix}
1 & 0 & 0 \\
0 & 3 & 0 \\
0 & 0 & 0 \\
\end{bmatrix} \quad
\begin{bmatrix}
0 & 0 & 0 \\
0 & 0 & 0 \\
0 & 0 & 7 \\
\end{bmatrix}
$
，可以发现，任何三阶对角矩阵均可用这三个矩阵的线性组合生成，因此，他们生成了三阶对角矩阵空间，即这三个矩阵是三阶对角矩阵空间的一组基。
