# 博弈论基础知识

## 正规形式博弈

**定义1.1：** 一个正规形式博弈由以下部分构成： 

- 玩家(player)的人数$n$，我们也将所有玩家记为$[n]=\begin{Bmatrix}1,2,\dots,n\end{Bmatrix}$；
- 对于其中的每一个玩家$i\epsilon [n]$，我们记其可以使用的策略(也称为纯策略)集合为$S_ i$；
- 还需要考虑的其收益函数$u_ i:\times _ {j\epsilon[n]}S_ j\to \mathbb{R}$，用以表示当给定所有玩家策略时玩家$i$的收益。

我们记正规形式博弈为$G = (n,\\{S\_i\\}\_\{i\epsilon [n]\},\\{u\_ {i}\\}\_\{i\epsilon [n]\})$。

## 混合策略等数学概念

**定义1.2：** 对于一个正规形式博弈$G=(n,\\{S\_i\\}\_\{i\epsilon[n]\},\\{u\_i\\}\_\{i\epsilon[n]\})$，我们定义：

- 在进行博弈前，每个玩家$i\epsilon[n]$从$S_i$中选择一个**纯策略**$s_i$，一起拼成策略向量$\mathbf{s}=\begin{Bmatrix}s_ 1,s_2,\dots,s_ n\end{Bmatrix}$，我们记$S:=\times_{i\epsilon [n]}S_i$为纯策略集合，所以我们有$\mathbf{s}\epsilon S$。
- 玩家$i$的**混合策略**包含于其策略集合$S_i$上的所有概率分布，记做：

$$
\Delta^{S_i}:=\begin{Bmatrix}\mathbf{x_i}\epsilon\mathbb{R}^{S_i}\vert\sum_{s_i\epsilon S_i}x_{i,s_i}=1且x_{i,s_i}\geq0,\forall s_i\epsilon S_i\end{Bmatrix}\tag{1.1}
$$

- 我们也将$\Delta=\times_{i\epsilon[n]} \Delta ^{S_ i}$称为**混合策略集合**，其中的元素$\mathbf{x}\epsilon\Delta$集合混合策略向量。
- 当玩家使用混合策略时，我们可以看成玩家$i\epsilon [n]$同时以$\Delta^{S_ i}$独立出来采用出一个纯策略，并以此计算个人的收益 。于是当我们给定混合策略向量$\mathbf{x}\epsilon \Delta$后，我们可以计算**期望收益**(Expected Payoff)为式(1.2)。其中$\mathbf{s}\sim\mathbf{x}$表示玩家$i\epsilon[n]$以混合策略$\mathbf{x}_ i\epsilon\Delta^{S_ i}$从$S_ i$中独立采样出$s_ i$后共同组成向量$\mathbf{s}\epsilon S$。


$$
\begin{equation}
u\_i(\mathbf{x}):=\mathbb{E}\_{\mathbf{s}\sim\mathbf{x}}[u\_i(\mathbf{s})]=\sum\_{\mathbf{s}\epsilon S}u\_ i(\mathbf{s})\prod\_{j\epsilon [n]}x\_{j,s\_j}\tag{1.2}
\end{equation}
$$



## 纳什均衡与纳什定理

**定义1.3纳什均衡：** 对于一个正规形式博弈$G=(n,\\{S\_i\\}\_\{i\epsilon [n]\},\\{u\_i\\}\_\{i\epsilon[n]\})$，一组混合策略向量$x\epsilon \Delta$是一个纳什均衡，当且仅当对于任意玩家$i\epsilon [n]$和任意策略${x_ i}'\epsilon \Delta^{S_ i}$，我们有：
$$
u_ i(x) \gt u_ i({x}'_ i,x_ {-i})\tag{1.3}
$$
其中$x_ {-i}$表示除玩家$i$之外其余玩家组成的混合策略向量。

通俗的来说，即是每个参与者都知道其他参与者的均衡策略的情况下，没有参与者可以通过改变自身策略使自己期望受益的一种策略组合。纳什均衡中常见的案例有：囚徒困境、智猪博弈、普通范式博弈、饿狮博弈。



**定理1.4纳什定理：** 当博弈中的玩家数是有限个，且每个玩家仅有有限个纯策略时，该博弈一定存在纳什均衡。



## 纳什均衡的计算复杂性





## 博弈论的两个分支

### 非合作博弈





### 合作博弈



## 参考文献

[1] [纳什均衡]([纳什均衡 - 维基百科，自由的百科全书 (wikipedia.org)](https://zh.wikipedia.org/wiki/纳什均衡))

[2] [三十分钟理解博弈论“纳什均衡” - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/41465296)

