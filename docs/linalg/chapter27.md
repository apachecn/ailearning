
# 第二十七讲：复数矩阵和快速傅里叶变换

本讲主要介绍复数向量、复数矩阵的相关知识（包括如何做复数向量的点积运算、什么是复数对称矩阵等），以及傅里叶矩阵（最重要的复数矩阵）和快速傅里叶变换。

## 复数矩阵运算

先介绍复数向量，我们不妨换一个字母符号来表示：$z=\begin{bmatrix}z_1\\z_2\\\vdots\\z_n\end{bmatrix}$，向量的每一个分量都是复数。此时$z$不再属于$\mathbb{R}^n$实向量空间，它现在处于$\mathbb{C}^n$复向量空间。

### 计算复向量的模

对比实向量，我们计算模只需要计算$\left|v\right|=\sqrt{v^Tv}$即可，而如果对复向量使用$z^Tz$则有$z^Tz=\begin{bmatrix}z_1&z_2&\cdots&z_n\end{bmatrix}\begin{bmatrix}z_1\\z_2\\\vdots\\z_n\end{bmatrix}=z_1^2+z_2^2+\cdots+z_n^2$，这里$z_i$是复数，平方后虚部为负，求模时本应相加的运算变成了减法。（如向量$\begin{bmatrix}1&i\end{bmatrix}$，右乘其转置后结果为$0$，但此向量的长度显然不是零。）

根据上一讲我们知道，应使用$\left|z\right|=\sqrt{\bar{z}^Tz}$，即$\begin{bmatrix}\bar z_1&\bar z_2&\cdots&\bar z_n\end{bmatrix}\begin{bmatrix}z_1\\z_2\\\vdots\\z_n\end{bmatrix}$，即使用向量共轭的转置乘以原向量即可。（如向量$\begin{bmatrix}1&i\end{bmatrix}$，右乘其共轭转置后结果为$\begin{bmatrix}1&-i\end{bmatrix}\begin{bmatrix}1\\i\end{bmatrix}=2$。）

我们把共轭转置乘以原向量记为$z^Hz$，$H$读作埃尔米特（人名为Hermite，形容词为Hermitian）

### 计算向量的内积

有了复向量模的计算公式，同理可得，对于复向量，内积不再是实向量的$y^Tx$形式，复向量内积应为$y^Hx$。

### 对称性

对于实矩阵，$A^T=A$即可表达矩阵的对称性。而对于复矩阵，我们同样需要求一次共轭$\bar{A}^T=A$。举个例子$\begin{bmatrix}2&3+i\\3-i&5\end{bmatrix}$是一个复数情况下的对称矩阵。这叫做埃尔米特矩阵，有性质$A^H=A$。

### 正交性

在第十七讲中，我们这样定义标准正交向量：$q_i^Tq_j=\begin{cases}0\quad i\neq j\\1\quad i=j\end{cases}$。现在，对于复向量我们需要求共轭：$\bar{q}_i^Tq_j=q_i^Hq_j=\begin{cases}0\quad i\neq j\\1\quad i=j\end{cases}$。

第十七讲中的标准正交矩阵：$Q=\Bigg[q_1\ q_2\ \cdots\ q_n\Bigg]$有$Q^TQ=I$。现在对于复矩阵则有$Q^HQ=I$。

就像人们给共轭转置起了个“埃尔米特”这个名字一样，正交性（orthogonal）在复数情况下也有了新名字，酉（unitary），酉矩阵（unitary matrix）与正交矩阵类似，满足$Q^HQ=I$的性质。而前面提到的傅里叶矩阵就是一个酉矩阵。

## 傅里叶矩阵

$n$阶傅里叶矩阵$F_n=\begin{bmatrix}1&1&1&\cdots&1\\1&w&w^2&\cdots&w^{n-1}\\1&w^2&w^4&\cdots&w^{2(n-1)}\\\vdots&\vdots&\vdots&\ddots&\vdots\\1&w^{n-1}&w^{2(n-1)}&\cdots&w^{(n-1)^2}\end{bmatrix}$，对于每一个元素有$(F_n)_{ij}=w^{ij}\quad i,j=0,1,2,\cdots,n-1$。矩阵中的$w$是一个非常特殊的值，满足$w^n=1$，其公式为$w=e^{i2\pi/n}$。易知$w$在复平面的单位圆上，$w=\cos\frac{2\pi}{n}+i\sin\frac{2\pi}{n}$。

在傅里叶矩阵中，当我们计算$w$的幂时，$w$在单位圆上的角度翻倍。比如在$6$阶情形下，$w=e^{2\pi/6}$，即位于单位圆上$60^\circ$角处，其平方位于单位圆上$120^\circ$角处，而$w^6$位于$1$处。从开方的角度看，它们是$1$的$6$个六次方根，而一次的$w$称为原根。

* 我们现在来看$4$阶傅里叶矩阵，先计算$w$有$w=i,\ w^2=-1,\ w^3=-i,\ w^4=1$，$F_4=\begin{bmatrix}1&1&1&1\\1&i&i^2&i^3\\1&i^2&i^4&i^6\\1&i^3&i^6&i^9\end{bmatrix}=\begin{bmatrix}1&1&1&1\\1&i&-1&-i\\1&-1&1&-1\\1&-i&-1&i\end{bmatrix}$。

    矩阵的四个列向量正交，我们验证一下第二列和第四列，$\bar{c_2}^Tc_4=1-0+1-0=0$，正交。不过我们应该注意到，$F_4$的列向量并不是标准的，我们可以给矩阵乘上系数$\frac{1}{2}$（除以列向量的长度）得到标准正交矩阵$F_4=\frac{1}{2}\begin{bmatrix}1&1&1&1\\1&i&-1&-i\\1&-1&1&-1\\1&-i&-1&i\end{bmatrix}$。此时有$F_4^HF_4=I$，于是该矩阵的逆矩阵也就是其共轭转置$F_4^H$。
    
## 快速傅里叶变换（Fast Fourier transform/FFT）

对于傅里叶矩阵，$F_6,\ F_3$、$F_8,\ F_4$、$F_{64},\ F_{32}$之间有着特殊的关系。

举例，有傅里叶矩阵$F_64$，一般情况下，用一个列向量右乘$F_{64}$需要约$64^2$次计算，显然这个计算量是比较大的。我们想要减少计算量，于是想要分解$F_{64}$，联系到$F_{32}$，有$\Bigg[F_{64}\Bigg]=\begin{bmatrix}I&D\\I&-D\end{bmatrix}\begin{bmatrix}F_{32}&0\\0&F_{32}\end{bmatrix}\begin{bmatrix}1&&\cdots&&&0&&\cdots&&\\0&&\cdots&&&1&&\cdots&&\\&1&\cdots&&&&0&\cdots&&\\&0&\cdots&&&&1&\cdots&&\\&&&\ddots&&&&&\ddots&&\\&&&\ddots&&&&&\ddots&&\\&&&\cdots&1&&&&\cdots&0\\&&&\cdots&0&&&&\cdots&1\end{bmatrix}$。

我们分开来看等式右侧的这三个矩阵：

* 第一个矩阵由单位矩阵$I$和对角矩阵$D=\begin{bmatrix}1&&&&\\&w&&&\\&&w^2&&\\&&&\ddots&\\&&&&w^{31}\end{bmatrix}$组成，我们称这个矩阵为修正矩阵，显然其计算量来自$D$矩阵，对角矩阵的计算量约为$32$即这个修正矩阵的计算量约为$32$，单位矩阵的计算量忽略不计。

* 第二个矩阵是两个$F_{32}$与零矩阵组成的，计算量约为$2\times 32^2$。

* 第三个矩阵通常记为$P$矩阵，这是一个置换矩阵，其作用是讲前一个矩阵中的奇数列提到偶数列之前，将前一个矩阵从$\Bigg[x_0\ x_1\ \cdots\Bigg]$变为$\Bigg[x_0\ x_2\ \cdots\ x_1\ x_3\ \cdots\Bigg]$，这个置换矩阵的计算量也可以忽略不计。（这里教授似乎在黑板上写错了矩阵，可以参考[FFT](https://math.berkeley.edu/~berlek/classes/CLASS.110/LECTURES/FFT)、[How the FFT is computed](http://vmm.math.uci.edu/ODEandCM/PDF_Files/FFT_Appendix_K.pdf)做进一步讨论。）

所以我们把$64^2$复杂度的计算化简为$2\times 32^2+32$复杂度的计算，我们可以进一步化简$F_{32}$得到与$F_{16}$有关的式子$\begin{bmatrix}I_{32}&D_{32}\\I_{32}&-D_{32}\end{bmatrix}\begin{bmatrix}I_{16}&D_{16}&&\\I_{16}&-D_{16}&&\\&&I_{16}&D_{16}\\&&I_{16}&-D_{16}\end{bmatrix}\begin{bmatrix}F_{16}&&&\\&F_{16}&&\\&&F_{16}&\\&&&F_{16}\end{bmatrix}\begin{bmatrix}P_{16}&\\&P_{16}\end{bmatrix}\Bigg[\ P_{32}\ \Bigg]$。而$32^2$的计算量进一步分解为$2\times 16^2+16$的计算量，如此递归下去我们最终得到含有一阶傅里叶矩阵的式子。

来看化简后计算量，$2\left(2\left(2\left(2\left(2\left(2\left(1\right)^2+1\right)+2\right)+4\right)+8\right)+16\right)+32$，约为$6\times 32=\log_264\times \frac{64}{2}$，算法复杂度为$\frac{n}{2}\log_2n$。

于是原来需要$n^2$的运算现在只需要$\frac{n}{2}\log_2n$就可以实现了。不妨看看$n=10$的情况，不使用FFT时需要$n^2=1024\times 1024$次运算，使用FFT时只需要$\frac{n}{2}\log_2n=5\times 1024$次运算，运算量大约是原来的$\frac{1}{200}$。

下一讲将继续介绍特征值、特征向量及正定矩阵。
