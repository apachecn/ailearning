
# 第十三讲：复习一

1. 令$u, v, w$是$\mathbb{R}^7$空间内的非零向量：则$u, v, w$生成的向量空间可能是$1, 2, 3$维的。

2. 有一个$5 \times 3$矩阵$U$，该矩阵为阶梯矩阵（echelon form），有$3$个主元：则能够得到该矩阵的秩为$3$，即三列向量线性无关，不存在非零向量使得三列的线性组合为零向量，所以该矩阵的零空间应为$\begin{bmatrix}0\\0\\0\\ \end{bmatrix}$。

3. 接上一问，有一个$10 \times 3$矩阵$B=\begin{bmatrix}U\\2U \end{bmatrix}$，则化为最简形式（阶梯矩阵）应为$\begin{bmatrix}U\\0 \end{bmatrix}$，$rank(B)=3$。

4. 接上一问，有一个矩阵型为$C=\begin{bmatrix}U & U \\ U & 0 \end{bmatrix}$，则化为最简形式应为$\begin{bmatrix}U & 0 \\ 0 & U \end{bmatrix}$，$rank(C)=6$。矩阵$C$为$10 \times 6$矩阵，$dim N(C^T)=m-r=4$。

5. 有$Ax=\begin{bmatrix}2\\4\\2\\ \end{bmatrix}$，并且$x=\begin{bmatrix}2\\0\\0\\ \end{bmatrix}+c\begin{bmatrix}1\\1\\0\\ \end{bmatrix}+d\begin{bmatrix}0\\0\\1 \end{bmatrix}$，则等号右侧$b$向量的列数应为$A$的行数，且解的列数应为$A$的列数，所以$A$是一个$3 \times 3$矩阵。从解的结构可知自由元有两个，则$rank(A)=1, dim N(A)=2$。从解的第一个向量得出，矩阵$A$的第一列是$\begin{bmatrix}1\\2\\1 \end{bmatrix}$；解的第二个向量在零空间中，说明第二列与第一列符号相反，所以矩阵第二列是$\begin{bmatrix}-1\\-2\\-1 \end{bmatrix}$；解的第三个向量在零空间中，说明第三列为零向量；综上，$A=\begin{bmatrix}1 & -1 & 0\\ 2 & -2 & 0\\ 1 & -1 & 0\\ \end{bmatrix}$。

6. 接上一问，如何使得$Ax=b$有解？即使$b$在矩阵$A$的列空间中。易知$A$的列空间型为$c\begin{bmatrix}1\\2\\1\\ \end{bmatrix}$，所以使$b$为向量$\begin{bmatrix}1\\2\\1\\ \end{bmatrix}$的倍数即可。

7. 有一方阵的零空间中只有零向量，则其左零空间也只有零向量。

8. 由$5 \times 5$矩阵组成的矩阵空间，其中的可逆矩阵能否构成子空间？两个可逆矩阵相加的结果并不一定可逆，况且零矩阵本身并不包含在可逆矩阵中。其中的奇异矩阵（singular matrix，非可逆矩阵）也不能组成子空间，因为其相加的结果并不一定能够保持不可逆。

9. 如果$B^2=0$，并不能得出$B=0$，反例：$\begin{bmatrix}0 & 1\\ 0 & 0\\ \end{bmatrix}$，**这个矩阵经常会被用作反例**。

10. $n \times n$矩阵的列向量线性无关，则是否$\forall b, Ax=b$有解？是的，因为方阵各列线性无关，所以方阵满秩，它是可逆矩阵，肯定有解。

11. 有
$
B=
\begin{bmatrix}
1 & 1 & 0 \\
0 & 1 & 0 \\
1 & 0 & 1 \\
\end{bmatrix}
\begin{bmatrix}
1 & 0 & -1 & 2 \\
0 & 1 & 1 & -1 \\
0 & 0 & 0 & 0 \\
\end{bmatrix}
$，在不解出$B$的情况下，求$B$的零空间。可以观察得出前一个矩阵是可逆矩阵，设$B=CD$，则求零空间$Bx=0, CDx=0$，而$C$是可逆矩阵，则等式两侧同时乘以$C^{-1}$有$C^{-1}CDx=Dx=0$，所以当$C$为可逆矩阵时，有$N(CD)=N(D)$，即左乘逆矩阵不会改变零空间。本题转化为求$D$的零空间，$N(B)$的基为
$\begin{bmatrix}-F\\I\\ \end{bmatrix}$，也就是$\begin{bmatrix}1\\-1\\1\\0 \end{bmatrix}\quad\begin{bmatrix}-2\\1\\0\\1\end{bmatrix}$

12. 接上题，求$Bx=\begin{bmatrix}1\\0\\1\\ \end{bmatrix}$的通解。观察$B=CD$，易得$B$矩阵的第一列为$\begin{bmatrix}1\\0\\1\\ \end{bmatrix}$，恰好与等式右边一样，所以$\begin{bmatrix}1\\0\\0\\0\\ \end{bmatrix}$可以作为通解中的特解部分，再利用上一问中求得的零空间的基，得到通解
$
x=
\begin{bmatrix}1\\0\\0\\0\\ \end{bmatrix}+
c_1\begin{bmatrix}1\\-1\\1\\0 \end{bmatrix}+c_2\begin{bmatrix}-2\\1\\0\\1\end{bmatrix}
$

13. 对于任意方阵，其行空间等于列空间？不成立，可以使用$\begin{bmatrix}0 & 1\\ 0 & 0\\ \end{bmatrix}$作为反例，其行空间是向量$\begin{bmatrix}0 & 1\\ \end{bmatrix}$的任意倍数，而列空间是向量$\begin{bmatrix}1 & 0\\ \end{bmatrix}$的任意倍数。但是如果该方阵是对称矩阵，则成立。

14. $A$与$-A$的四个基本子空间相同。

15. 如果$A, B$的四个基本子空间相同，则$A, B$互为倍数关系。不成立，如任意两个$n$阶可逆矩阵，他们的列空间、行空间均为$\mathbb{R}^n$，他们的零空间、左零空间都只有零向量，所以他们的四个基本子空间相同，但是并不一定具有倍数关系。

16. 如果交换矩阵的某两行，则其行空间与零空间保持不变，而列空间与左零空间均已改变。

17. 为什么向量$v=\begin{bmatrix}1\\2\\3 \end{bmatrix}$不能同时出现在矩阵的行空间与零空间中？令$A\begin{bmatrix}1\\2\\3 \end{bmatrix}=\begin{bmatrix}0\\0\\0 \end{bmatrix}$，很明显矩阵$A$中不能出现值为$\begin{bmatrix}1 & 2 & 3 \end{bmatrix}$的行向量，否则无法形成等式右侧的零向量。这里引入正交（perpendicular）的概念，矩阵的行空间与零空间正交，它们仅共享零向量。
