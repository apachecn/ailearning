
# 第四讲：$A$ 的 $LU$ 分解

$AB$的逆矩阵：
$$
\begin{aligned}
A \cdot A^{-1} = I & = A^{-1} \cdot A\\
(AB) \cdot (B^{-1}A^{-1}) & = I\\
\textrm{则} AB \textrm{的逆矩阵为} & B^{-1}A^{-1}
\end{aligned}
$$

$A^{T}$的逆矩阵：
$$
\begin{aligned}
(A \cdot A^{-1})^{T} & = I^{T}\\
(A^{-1})^{T} \cdot A^{T} & = I\\
\textrm{则} A^{T} \textrm{的逆矩阵为} & (A^{-1})^{T}
\end{aligned}
$$

## 将一个 $n$ 阶方阵 $A$ 变换为 $LU$ 需要的计算量估计：

1. 第一步，将$a_{11}$作为主元，需要的运算量约为$n^2$
$$
\begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{n1} & a_{n2} & \cdots & a_{nn} \\
\end{bmatrix}
\underrightarrow{消元}
\begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
0      & a_{22} & \cdots & a_{2n} \\
0      & \vdots & \ddots & \vdots \\
0      & a_{n2} & \cdots & a_{nn} \\
\end{bmatrix}
$$

2. 以此类推，接下来每一步计算量约为$(n-1)^2、(n-2)^2、\cdots、2^2、1^2$。

3. 则将 $A$ 变换为 $LU$ 的总运算量应为$O(n^2+(n-1)^2+\cdots+2^2+1^2)$，即$O(\frac{n^3}{3})$。

置换矩阵(Permutation Matrix)：

3阶方阵的置换矩阵有6个：
$$
\begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1 \\
\end{bmatrix}
\begin{bmatrix}
0 & 1 & 0 \\
1 & 0 & 0 \\
0 & 0 & 1 \\
\end{bmatrix}
\begin{bmatrix}
0 & 0 & 1 \\
0 & 1 & 0 \\
1 & 0 & 0 \\
\end{bmatrix}
\begin{bmatrix}
1 & 0 & 0 \\
0 & 0 & 1 \\
0 & 1 & 0 \\
\end{bmatrix}
\begin{bmatrix}
0 & 1 & 0 \\
0 & 0 & 1 \\
1 & 0 & 0 \\
\end{bmatrix}
\begin{bmatrix}
0 & 0 & 1 \\
1 & 0 & 0 \\
0 & 1 & 0 \\
\end{bmatrix}
$$

$n$阶方阵的置换矩阵有$\binom{n}{1}=n!$个。
