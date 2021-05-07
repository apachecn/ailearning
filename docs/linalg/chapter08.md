
# 第八讲：求解$Ax=b$：可解性和解的结构

举例，同上一讲：$3 \times 4$矩阵
$
A=
\begin{bmatrix}
1 & 2 & 2 & 2\\
2 & 4 & 6 & 8\\
3 & 6 & 8 & 10\\
\end{bmatrix}
$，求$Ax=b$的特解：

写出其增广矩阵（augmented matrix）$\left[\begin{array}{c|c}A & b\end{array}\right]$：

$$
\left[
\begin{array}{c c c c|c}
1 & 2 & 2 & 2 & b_1 \\
2 & 4 & 6 & 8 & b_2 \\
3 & 6 & 8 & 10 & b_3 \\
\end{array}
\right]
\underrightarrow{消元}
\left[
\begin{array}{c c c c|c}
1 & 2 & 2 & 2 & b_1 \\
0 & 0 & 2 & 4 & b_2-2b_1 \\
0 & 0 & 0 & 0 & b_3-b_2-b_1 \\
\end{array}
\right]
$$

显然，有解的必要条件为$b_3-b_2-b_1=0$。

讨论$b$满足什么条件才能让方程$Ax=b$有解（solvability condition on b）：当且仅当$b$属于$A$的列空间时。另一种描述：如果$A$的各行线性组合得到$0$行，则$b$端分量做同样的线性组合，结果也为$0$时，方程才有解。

解法：令所有自由变量取$0$，则有$
\Big\lbrace
\begin{eqnarray*}
x_1 & + & 2x_3 & = & 1 \\
    &   & 2x_3 & = & 3 \\
\end{eqnarray*}
$
，解得
$
\Big\lbrace
\begin{eqnarray*}
x_1 & = & -2 \\
x_3 & = & \frac{3}{2} \\
\end{eqnarray*}
$
，代入$Ax=b$求得特解
$
x_p=
\begin{bmatrix}
-2 \\ 0 \\ \frac{3}{2} \\ 0
\end{bmatrix}
$。

令$Ax=b$成立的所有解：

$$
\Big\lbrace
\begin{eqnarray}
A & x_p & = & b \\
A & x_n & = & 0 \\
\end{eqnarray}
\quad
\underrightarrow{两式相加}
\quad
A(x_p+x_n)=b
$$

即$Ax=b$的解集为其特解加上零空间，对本例有：
$
x_{complete}=
\begin{bmatrix}
-2 \\ 0 \\ \frac{3}{2} \\ 0
\end{bmatrix}
+
c_1\begin{bmatrix}-2\\1\\0\\0\\\end{bmatrix}
+
c_2\begin{bmatrix}2\\0\\-2\\1\\\end{bmatrix}
$

对于$m \times n$矩阵$A$，有矩阵$A$的秩$r \leq min(m, n)$

列满秩$r=n$情况：
$
A=
\begin{bmatrix}
1 & 3 \\
2 & 1 \\
6 & 1 \\
5 & 1 \\
\end{bmatrix}
$
，$rank(A)=2$，要使$Ax=b, b \neq 0$有非零解，$b$必须取$A$中各列的线性组合，此时A的零空间中只有$0$向量。

行满秩$r=m$情况：
$
A=
\begin{bmatrix}
1 & 2 & 6 & 5 \\
3 & 1 & 1 & 1 \\
\end{bmatrix}
$
，$rank(A)=2$，$\forall b \in R^m都有x \neq 0的解$，因为此时$A$的列空间为$R^m$，$b \in R^m$恒成立，组成$A$的零空间的自由变量有n-r个。

行列满秩情况：$r=m=n$，如
$
A=
\begin{bmatrix}
1 & 2 \\
3 & 4 \\
\end{bmatrix}
$
，则$A$最终可以化简为$R=I$，其零空间只包含$0$向量。

总结：

$$\begin{array}{c|c|c|c}r=m=n&r=n\lt m&r=m\lt n&r\lt m,r\lt n\\R=I&R=\begin{bmatrix}I\\0\end{bmatrix}&R=\begin{bmatrix}I&F\end{bmatrix}&R=\begin{bmatrix}I&F\\0&0\end{bmatrix}\\1\ solution&0\ or\ 1\ solution&\infty\ solution&0\ or\ \infty\ solution\end{array}$$
