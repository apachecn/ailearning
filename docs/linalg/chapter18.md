
# 第十八讲：行列式及其性质

本讲我们讨论出行列式（determinant）的性质：

1. $\det{I}=1$，单位矩阵行列式值为一。
2. 交换行行列式变号。

    在给出第三个性质之前，先由前两个性质可知，对置换矩阵有$\det P=\begin{cases}1\quad &even\\-1\quad &odd\end{cases}$。

    举例：$\begin{vmatrix}1&0\\0&1\end{vmatrix}=1,\quad\begin{vmatrix}0&1\\1&0\end{vmatrix}=-1$，于是我们猜想，对于二阶方阵，行列式的计算公式为$\begin{vmatrix}a&b\\c&d\end{vmatrix}=ad-bc$。

3. a. $\begin{vmatrix}ta&tb\\tc&td\end{vmatrix}=t\begin{vmatrix}a&b\\c&d\end{vmatrix}$。

    b. $\begin{vmatrix}a+a'&b+b'\\c&d\end{vmatrix}=\begin{vmatrix}a&b\\c&d\end{vmatrix}+\begin{vmatrix}a'&b'\\c&d\end{vmatrix}$。
    
    **注意**：~~这里并不是指$\det (A+B)=\det A+\det B$，方阵相加会使每一行相加，这里仅是针对某一行的线性变换。~~

4. 如果两行相等，则行列式为零。使用性质2交换两行易证。
5. 从第$k$行中减去第$i$行的$l$倍，行列式不变。这条性质是针对消元的，我们可以先消元，将方阵变为上三角形式后再计算行列式。

    举例：$\begin{vmatrix}a&b\\c-la&d-lb\end{vmatrix}\stackrel{3.b}{=}\begin{vmatrix}a&b\\c&d\end{vmatrix}+\begin{vmatrix}a&b\\-la&-lb\end{vmatrix}\stackrel{3.a}{=}\begin{vmatrix}a&b\\c&d\end{vmatrix}-l\begin{vmatrix}a&b\\a&b\end{vmatrix}\stackrel{4}{=}\begin{vmatrix}a&b\\c&d\end{vmatrix}$

6. 如果方阵的某一行为零，则其行列式值为零。使用性质3.a对为零行乘以不为零系数$l$，使$l\det A=\det A$即可证明；或使用性质5将某行加到为零行，使存在两行相等后使用性质4即可证明。

7. 有上三角行列式$U=\begin{vmatrix}d_{1}&*&\cdots&*\\0&d_{2}&\cdots&*\\\vdots&\vdots&\ddots&\vdots\\0&0&\cdots&d_{n}\end{vmatrix}$，则$\det U=d_1d_2\cdots d_n$。使用性质5，从最后一行开始，将对角元素上方的$*$元素依次变为零，可以得到型为$D=\begin{vmatrix}d_{1}&0&\cdots&0\\0&d_{2}&\cdots&0\\\vdots&\vdots&\ddots&\vdots\\0&0&\cdots&d_{n}\end{vmatrix}$的对角行列式，再使用性质3将对角元素提出得到$d_nd_{n-1}\cdots d_1\begin{vmatrix}1&0&\cdots&0\\0&1&\cdots&0\\\vdots&\vdots&\ddots&\vdots\\0&0&\cdots&1\end{vmatrix}$，得证。

8. 当矩阵$A$为奇异矩阵时，$\det A=0$；当且仅当$A$可逆时，有$\det A\neq0$。如果矩阵可逆，则化简为上三角形式后各行都含有主元，行列式即为主元乘积；如果矩阵奇异，则化简为上三角形式时会出现全零行，行列式为零。

    再回顾二阶情况：$\begin{vmatrix}a&b\\c&d\end{vmatrix}\xrightarrow{消元}\begin{vmatrix}a&b\\0&d-\frac{c}{a}b\end{vmatrix}=ad-bc$，前面的猜想得到证实。

9. $\det AB=(\det A)(\det B)$。使用这一性质，$\det I=\det{A^{-1}A}=\det A^{-1}\det A$，所以$\det A^{-1}=\frac{1}{\det A}$。

    同时还可以得到：$\det A^2=(\det A)^2$，以及$\det 2A=2^n\det A$，这个式子就像是求体积，对三维物体有每边翻倍则体积变为原来的八倍。

10. $\det A^T=\det A$，前面一直在关注行的属性给行列式带来的变化，有了这条性质，行的属性同样适用于列，比如对性质2就有“交换列行列式变号”。
    
    证明：$\left|A^T\right|=\left|A\right|\rightarrow\left|U^TL^T\right|=\left|LU\right|\rightarrow\left|U^T\right|\left|L^T\right|=\left|L\right|\left|U\right|$，值得注意的是，$L, U$的行列式并不因为转置而改变，得证。
