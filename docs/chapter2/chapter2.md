# 强化学习简介

## 强化学习框架

### 马尔可夫决策过程(MDP) 

**定义2.1：** 马尔可夫决策过程可被元祖$\langle \mathbb{S},\mathbb{A},P,R,\gamma \rangle$中关键元素所描述。

- $\mathbb{S}：$ 环境状态集合。
- $\mathbb{A}：$ 智能体可能的动作集合。
- $P:\mathbb{S}\times\mathbb{A}\to\Delta(\mathbb{S})：$ 在时刻$t$时，基于智能体的动作$a\epsilon\mathbb{A}$，在下一时刻，环境由状态$s\epsilon\mathbb{S}$转换到${s}'\epsilon\mathbb{S}$的概率。
- $R:\mathbb{S}\times\mathbb{A}\times\mathbb{S}\to\mathbb{R}：$ 表示奖励函数，即智能体采取动作$a$后，环境从状态$s$转换到状态${s}'$，返回一个标量值。奖励的绝对值有界于$R_{max}$。
- $\gamma\epsilon[0,1]：$ 折扣系数。

智能体的目标是：解马尔可夫决策过程，也即寻找到最大化奖励的**最优策略**。在数学上，智能体的目标是找到一个马尔可夫性的(Markovian)和静态的(Stationary)策略。其中，Markovian是指输入仅仅取决于当前状态，Stationary是指函数形式与时间独立，即策略函数为$\pi:\mathbb{S}\to\Delta(\mathbb{A})$，$\Delta(\cdot)$指的是概率单纯形。如式2.1所示，最优策略引导智能体采取序列行动，从而最大化折扣累计回报。
$$
\begin{equation}
\mathbb{E}\_{s\_{t+1}\sim P(\cdot|s\_t,a\_t)}\begin{bmatrix}\sum\_{t\ge0}\gamma ^{t}R(s\_t,a\_t,s\_{t+1})\vert a\_t\sim\pi(\cdot\vert s\_t),s\_0\end{bmatrix}\tag{2.1}
\end{equation}
$$
在有限窗口中，智能体可采取非确定性策略，即策略函数是动作关于状态的概率分布；在无限窗口中，智能体可采取确定性策略，即策略函数是动作关于状态的函数。因此，在有限窗口的情景下，智能体的目标是寻找期望奖励最大化。其中，有限窗口或无限窗口指的是一个回合中智能体可与环境交互的时间步数。

在策略$\pi$下，基于目标函数(2.1)，可以定义状态-动作函数(Q函数)和价值函数为：
$$
\begin{equation}
Q^{\pi}(s,a)=\mathbb{E}^{\pi}\begin{bmatrix}\sum\_{t\ge0}\gamma^tR(s\_t,a\_t,s\_{t+1}\vert a\_0=a,s\_0=s)\end{bmatrix},\forall s\epsilon \mathbb{S},a\epsilon \mathbb{A}\tag{2.2}
\end{equation}
$$

$$
\begin{equation}
V^{\pi}(s)=\mathbb{E}^{\pi}\begin{bmatrix}\sum\_{t\ge0}\gamma^tR(s\_t,a\_t,s\_{t+1}\vert s\_0=s)\end{bmatrix},\forall s\epsilon \mathbb{S}\tag{2.3}
\end{equation}
$$

价值函数与Q函数，有如下关系：

$V^{\pi}(s)=\mathbb{E}\_{a\sim\pi(\cdot\vert s)}\begin{bmatrix}Q^{\pi}(s,a)\end{bmatrix}$

$Q^{\pi}=\mathbb{E}\_{{s}'\sim \pi(\cdot\vert s)}\begin{bmatrix}R(s,a,{s}')+V^{\pi}({s}')\end{bmatrix}$

 对于智能体来说，只要奖励函数和状态转换函数均具有马尔可夫性和静态性，那么最优策略一定存在。



### 部分可观测马尔可夫决策过程(POMDP)

若智能体不能精确的获取环境状态，只能根据**观测函数**获取真实状态的一个观测，那么该环境为部分可观测。马尔可夫决策过程适用于环境状态可精确获取的场景，而部分马尔可夫决策过程适用于部分可观测环境。其中，部分可观测马尔可夫决策过程(POMDP)为马尔可夫决策过程(MDP)的扩展版本。

**定义2.2：** 部分可观测马尔可夫决策过程可被元祖$\langle \mathbb{S},\mathbb{A},P,R,\gamma,\mathbb{O},O \rangle$所描述。除了MDP定义中元素之外，其余的元素含义如下：

- $\mathbb{O}：$ 智能体的观测集合。
- $O:\mathbb{S}\times\mathbb{A}\to\Delta(\mathbb{O})：$ 表示的是观测函数$O(o\vert a,{s}')$ ，给定动作$a\epsilon \mathbb{A}$和环境状态转换后的新状态${s}'\epsilon \mathbb{S}$，观测$o\epsilon \mathbb{O}$的概率分布。

智能体的策略变为$\pi:\mathbb{O}\to\Delta(\mathbb{A})$。

## 马尔可夫决策过程的解方法

### 基于值的方法



### 基于策略的方法



## 参考文献

[1] Yang Y ,  Wang J . An Overview of Multi-Agent Reinforcement Learning from Game Theoretical Perspective[J].  2020.
