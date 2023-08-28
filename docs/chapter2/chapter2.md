# 强化学习简介

与有监督学习和无监督学习不一样，强化学习要解决的是序列决策问题，需要不断的试错，从而智能体学习到最优策略，使其平均收益最大化。

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

#### DQN

DQN为深度强化学习的开篇之作，该算法基于深度学习作为Q-Learning的函数近似器，从而形成Deep Q Learning。在DQN论文中，表明，强化学习理论深深根植于动物行为的心理学和神经科学。研究表明，人类和其它动物是基于强化学习和层级传感器系统处理复杂问题。研究表明，多巴胺释放神经相位信号与时序差分强化学习算法类似。把深度学习用作Q-Learning算法的函数近似器，往往会产生学习不稳定的现象，而产生这种现象主要由状态观测之间的相关性和Q值与目标值之间的相关性造成。为了解决观测序列造成学习不稳定的现象，基于**经验回放**方法随机化数据，从而移除观测数据之间的相关性和平滑数据分布；为了解决Q值与目标值之间的相关性，对于学习网络迭代方式更新，而目标网络**周期性**更新。

#### Double DQN

若Q-Learning算法因环境的噪声而对状态-动作(s,a)估计有误差，那么很有可能产生高估的现象，这是因为Q-Learning中目标值为当前状态下的最大Q值。然而，高估并不意味着对模型训练有害处，因为高估是智能体对不确定性的乐观。若面对确定性的高估，会阻止智能体对最优策略的学习。

根据前人研究，Double Q-Learning通过分解Q函数目标值中动作的选择和状态-动作估计，可见式(1)，可以解决Q-learning的高估问题。
$$
Y_t=R_{t+1}+\gamma Q(S_{t+1},argmax_{a}Q(S_{t+1},a;\theta_{t}){\theta}'_{t})
$$
这种分解既满足了Q-Learning中贪心选择动作，又降低了高估的可能性。

那么，为了降低DQN对状态-动作对的高估，可基于深度学习作为函数拟合器，构造Double DQN。

#### Dueling Network

相较于DQN、Double DQN、Prioritized Replay，Dueling Network是强化学习网络架构上的创新。在实践中，发现，Dueling Network算法的表现性能是最好的。Dueling Network网络架构由两个部分组成，分别为状态价值![img](https://cdn.nlark.com/yuque/__latex/283b1b0d0929bc6fe1f092901d366e1a.svg)预测部分和优势函数![img](https://cdn.nlark.com/yuque/__latex/450c50a85770bd2a9af629098326981d.svg)预测部分，以上两个部分共享一个卷积模块，用于特征学习。该网络架构的提出是基于并不是所有状态下都需要知道动作的价值。

#### Rainbow



### 基于策略的方法



## 参考文献

[1] Yang Y ,  Wang J . An Overview of Multi-Agent Reinforcement Learning from Game Theoretical Perspective[J].  2020.
