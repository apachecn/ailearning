
# 第二十五讲：复习二

* 我们学习了正交性，有矩阵$Q=\Bigg[q_1\ q_2\ \cdots\ q_n\Bigg]$，若其列向量相互正交，则该矩阵满足$Q^TQ=I$。
* 进一步研究投影，我们了解了Gram-Schmidt正交化法，核心思想是求法向量，即从原向量中减去投影向量$E=b-P, P=Ax=\frac{A^Tb}{A^TA}\cdot A$。
* 接着学习了行列式，根据行列式的前三条性质，我们拓展出了性质4-10。
* 我们继续推导出了一个利用代数余子式求行列式的公式。
* 又利用代数余子式推导出了一个求逆矩阵的公式。
* 接下来我们学习了特征值与特征向量的意义：$Ax=\lambda x$，进而了解了通过$\det(A-\lambda I)=0$求特征值、特征向量的方法。
* 有了特征值与特征向量，我们掌握了通过公式$AS=\Lambda S$对角化矩阵，同时掌握了求矩阵的幂$A^k=S\Lambda^kS^{-1}$。

微分方程不在本讲的范围内。下面通过往年例题复习上面的知识。

1. *求$a=\begin{bmatrix}2\\1\\2\end{bmatrix}$的投影矩阵$P$*：$\Bigg($由$a\bot(b-p)\rightarrow A^T(b-A\hat x)=0$得到$\hat x=\left(A^TA\right)^{-1}A^Tb$，求得$p=A\hat x=A\left(A^TA\right)^{-1}A^Tb=Pb$最终得到$P\Bigg)$$\underline{P=A\left(A^TA\right)^{-1}A^T}\stackrel{a}=\frac{aa^T}{a^Ta}=\frac{1}{9}\begin{bmatrix}4&2&4\\2&1&2\\4&2&4\end{bmatrix}$。
    
    *求$P$矩阵的特征值*：观察矩阵易知矩阵奇异，且为秩一矩阵，则其零空间为$2$维，所以由$Px=0x$得出矩阵的两个特征向量为$\lambda_1=\lambda_2=0$；而从矩阵的迹得知$trace(P)=1=\lambda_1+\lambda_2+\lambda_3=0+0+1$，则第三个特征向量为$\lambda_3=1$。
    
    *求$\lambda_3=1$的特征向量*：由$Px=x$我们知道经其意义为，$x$过矩阵$P$变换后不变，又有$P$是向量$a$的投影矩阵，所以任何向量经过$P$变换都会落在$a$的列空间中，则只有已经在$a$的列空间中的向量经过$P$的变换后保持不变，即其特征向量为$x=a=\begin{bmatrix}2\\1\\2\end{bmatrix}$，也就是$Pa=a$。
    
    *有差分方程$u_{k+1}=Pu_k,\ u_0=\begin{bmatrix}9\\9\\0\end{bmatrix}$，求解$u_k$*：我们先不急于解出特征值、特征向量，因为矩阵很特殊（投影矩阵）。首先观察$u_1=Pu_0$，式子相当于将$u_0$投影在了$a$的列空间中，计算得$u_1=a\frac{a^Tu_0}{a^Ta}=3a=\begin{bmatrix}6\\3\\6\end{bmatrix}$（这里的$3$相当于做投影时的系数$\hat x$），其意义为$u_1$在$a$上且距离$u_0$最近。再来看看$u_2=Pu_1$，这个式子将$u_1$再次投影到$a$的列空间中，但是此时的$u_1$已经在该列空间中了，再次投影仍不变，所以有$u_k=P^ku_0=Pu_0=\begin{bmatrix}6\\3\\6\end{bmatrix}$。
    
    上面的解法利用了投影矩阵的特殊性质，如果在一般情况下，我们需要使用$AS=S\Lambda\rightarrow A=S\Lambda S^{-1} \rightarrow u_{k+1}=Au_k=A^{k+1}u_0, u_0=Sc\rightarrow u_{k+1}=S\Lambda^{k+1}S^{-1}Sc=S\Lambda^{k+1}c$，最终得到公式$A^ku_0=c_1\lambda_1^kx_1+c_2\lambda_2^kx_2+\cdots+c_n\lambda_n^kx_n$。题中$P$的特殊性在于它的两个“零特征值”及一个“一特征值”使得式子变为$A^ku_0=c_3x_3$，所以得到了上面结构特殊的解。
    
2. *将点$(1,4),\ (2,5),\ (3,8)$拟合到一条过零点的直线上*：设直线为$y=Dt$，写成矩阵形式为$\begin{bmatrix}1\\2\\3\end{bmatrix}D=\begin{bmatrix}4\\5\\8\end{bmatrix}$，即$AD=b$，很明显$D$不存在。利用公式$A^TA\hat D=A^Tb$得到$14D=38,\ \hat D=\frac{38}{14}$，即最佳直线为$y=\frac{38}{14}t$。这个近似的意义是将$b$投影在了$A$的列空间中。

3. *求$a_1=\begin{bmatrix}1\\2\\3\end{bmatrix}\ a_2=\begin{bmatrix}1\\1\\1\end{bmatrix}$的正交向量*：找到平面$A=\Bigg[a_1,a_2\Bigg]$的正交基，使用Gram-Schmidt法，以$a_1$为基准，正交化$a_2$，也就是将$a_2$中平行于$a_1$的分量去除，即$a_2-xa_1=a_2-\frac{a_1^Ta_2}{a_1^Ta_1}a_1=\begin{bmatrix}1\\1\\1\end{bmatrix}-\frac{6}{14}\begin{bmatrix}1\\2\\3\end{bmatrix}$。

4. *有$4\times 4$矩阵$A$，其特征值为$\lambda_1,\lambda_2,\lambda_3,\lambda_4$，则矩阵可逆的条件是什么*：矩阵可逆，则零空间中只有零向量，即$Ax=0x$没有非零解，则零不是矩阵的特征值。

    *$\det A^{-1}$是什么*：$\det A^{-1}=\frac{1}{\det A}$，而$\det A=\lambda_1\lambda_2\lambda_3\lambda_4$，所以有$\det A^{-1}=\frac{1}{\lambda_1\lambda_2\lambda_3\lambda_4}$。
    
    *$trace(A+I)$的迹是什么*：我们知道$trace(A)=a_{11}+a_{22}+a_{33}+a_{44}=\lambda_1+\lambda_2+\lambda_3+\lambda_4$，所以有$trace(A+I)=a_{11}+1+a_{22}+1+a_{33}+1+a_{44}+1=\lambda_1+\lambda_2+\lambda_3+\lambda_4+4$。
    
5. *有矩阵$A_4=\begin{bmatrix}1&1&0&0\\1&1&1&0\\0&1&1&1\\0&0&1&1\end{bmatrix}$，求$D_n=?D_{n-1}+?D_{n-2}$*：求递归式的系数，使用代数余子式将矩阵安第一行展开得$\det A_4=1\cdot\begin{vmatrix}1&1&0\\1&1&1\\0&1&1\end{vmatrix}-1\cdot\begin{vmatrix}1&1&0\\0&1&1\\0&1&1\end{vmatrix}=1\cdot\begin{vmatrix}1&1&0\\1&1&1\\0&1&1\end{vmatrix}-1\cdot\begin{vmatrix}1&1\\1&1\end{vmatrix}=\det A_3-\det A_2$。则可以看出有规律$D_n=D_{n-1}-D_{n-2}, D_1=1, D_2=0$。

    使用我们在差分方程中的知识构建方程组$\begin{cases}D_n&=D_{n-1}-D_{n-2}\\D_{n-1}&=D_{n-1}\end{cases}$，用矩阵表达有$\begin{bmatrix}D_n\\D_{n-1}\end{bmatrix}=\begin{bmatrix}1&-1\\1&0\end{bmatrix}\begin{bmatrix}D_{n-1}\\D_{n-2}\end{bmatrix}$。计算系数矩阵$A_c$的特征值，$\begin{vmatrix}1-\lambda&1\\1&-\lambda\end{vmatrix}=\lambda^2-\lambda+1=0$，解得$\lambda_1=\frac{1+\sqrt{3}i}{2},\lambda_2=\frac{1-\sqrt{3}i}{2}$，特征值为一对共轭复数。
    
    要判断递归式是否收敛，需要计算特征值的模，即实部平方与虚部平方之和$\frac{1}{4}+\frac{3}{4}=1$。它们是位于单位圆$e^{i\theta}$上的点，即$\cos\theta+i\sin\theta$，从本例中可以计算出$\theta=60^\circ$，也就是可以将特征值写作$\lambda_1=e^{i\pi/3},\lambda_2=e^{-i\pi/3}$。注意，从复平面单位圆上可以看出，这些特征值的六次方将等于一：$e^{2\pi i}=e^{2\pi i}=1$。继续深入观察这一特性对矩阵的影响，$\lambda_1^6=\lambda^6=1$，则对系数矩阵有$A_c^6=I$。则系数矩阵$A_c$服从周期变化，既不发散也不收敛。 

6. *有这样一类矩阵$A_4=\begin{bmatrix}0&1&0&0\\1&0&2&0\\0&2&0&3\\0&0&3&0\end{bmatrix}$，求投影到$A_3$列空间的投影矩阵*：有$A_3=\begin{bmatrix}0&1&0\\1&0&2\\0&2&0\end{bmatrix}$，按照通常的方法求$P=A\left(A^TA\right)A^T$即可，但是这样很麻烦。我们可以考察这个矩阵是否可逆，因为如果可逆的话，$\mathbb{R}^4$空间中的任何向量都会位于$A_4$的列空间，其投影不变，则投影矩阵为单位矩阵$I$。所以按行展开求行列式$\det A_4=-1\cdot-1\cdot-3\cdot-3=9$，所以矩阵可逆，则$P=I$。

    *求$A_3$的特征值及特征向量*：$\left|A_3-\lambda I\right|=\begin{vmatrix}-\lambda&1&0\\1&-\lambda&2\\0&2&-\lambda\end{vmatrix}=-\lambda^3+5\lambda=0$，解得$\lambda_1=0,\lambda_2=\sqrt 5,\lambda_3=-\sqrt 5$。
    
    我们可以猜测这一类矩阵的规律：奇数阶奇异，偶数阶可逆。
