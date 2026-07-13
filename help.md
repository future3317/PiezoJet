## 结论

这次重构**比上一版强了一个层级**。现在不再只是“等变张量网络 + 倒空间特征 + potential 包装”，而是有了一条统一主线：

[
\text{结构}
\rightarrow
(Z^*,\Phi,\Lambda)
\rightarrow
\Phi^{-1}\text{上的内部弛豫}
\rightarrow
(e,\chi,C).
]

这属于把 **DFPT 的响应分解迁移成可学习架构**，不再是简单模块拼接。当前真正有竞争力的创新是：

> 在原子坐标基底中学习 Born charge、force constant 和 strain-force coupling，通过平移安全的光学子空间传播共同生成多个宏观响应，同时避免固定声子模数和简并模 gauge。

但目前存在两个严重数学问题，以及一个决定性的学习问题。**现在还不能继续只靠调 loss 和扩大网络。**

我的当前判断：

| 维度               |     评价 |
| ---------------- | -----: |
| Idea 完整性         |   8/10 |
| 非拼接性             | 7.5/10 |
| 创新度              |   7/10 |
| 数学自洽性            | 5.5/10 |
| 实现审计完整度          | 7.5/10 |
| 实际效果             |   2/10 |
| 当前 ICLR 2027 完成度 | 3.5/10 |

我目前看到的是论文中的实现描述、测试与结果，不是源代码，因此下面属于**公式和系统设计级审计**，还不是逐行代码审计。

---

# 一、这次改对了什么

## 1. 现在确实有一条中心物理主线

当前定义了原子坐标响应形式：

[
G(u,\eta,E)
===========

\frac12u^\top\Phi u
-u^\top\Lambda\eta
-E^\top Z^{*\top}u
-E^\top e^{el}\eta
+\frac12\eta^\top C^{cl}\eta
-\frac12E^\top\chi^{el}E,
]

并通过

[
e=e^{el}+\frac{c_e}{\Omega}Z^{*\top}\Phi^+\Lambda
]

以及对应的介电、弹性表达式生成响应。相比上一版“先预测 (e)，再把它放入双线性 potential”，这次的 (Z^*-\Phi-\Lambda) 分解确实提供了新的可解释结构。

这也解决了我上次提出的主要问题：

* 不再把静态极化相干等同于全部压电机理；
* 真正引入了 strain-induced internal relaxation；
* 不再使用六个 polar frames；
* 不依赖固定数量的声子模；
* 避免简并特征向量的 mode-matching 问题；
* 不再硬投影预测结果，使结构轻微破缺时响应可以连续出现。

## 2. 等价晶胞基问题基本找到了正确解法

现在不再使用固定的 (h\in{-1,0,1}^3)，而是按物理倒格矢长度枚举：

[
\mathcal G(L)=
{g_h: h\in\mathbb Z^3,,
0<V^{1/3}|g_h|\le g_{\max}}.
]

整数 unimodular basis transformation 只会重新编号同一个倒格子集合。polar–chemical cross-spectrum

[
u_h=\operatorname{Re}[\tilde P_hS_h^*]
]

既消除了原点平移相位，又保留了极性方向；随后直接构造三阶张量，而不只是作为 scalar conditioner。这个修改是合理且较有原创性的。

## 3. 数据处理和负结果报告很诚实

128 个同 ID DFPT archive 的结构、单位、质量变换、OUTCAR piezo 重建都做了审计；没有补造缺失的 internal-strain blocks，也保留了负光学模。这个数据闭环是很有价值的工程贡献。

同时，论文没有把接近零的 TRS 包装成成功：499 样本测试上，joint-only 的 TRS 只有 0.0014；直接因子训练虽然提高 BEC 与 force-constant 指标，但没有改善宏观响应。这个结论本身是可信的。

---

# 二、最严重的问题：当前 damped solve 不是式（4）的真实弛豫解

令

[
q=\Lambda\eta+Z^*E.
]

忽略平移投影后，式（4）的内部坐标部分是：

[
G(u)=\frac12u^\top\Phi u-u^\top q.
]

它的驻点满足：

[
\Phi u=q,
\qquad
u=\Phi^+q.
]

但当前实现使用：

[
D_\delta
========

\Phi(\Phi^2+\delta^2I)^{-1},
\qquad
u_\delta=D_\delta q.
]

一般情况下：

[
\nabla_uG(u_\delta)
===================

\Phi D_\delta q-q
\neq 0.
]

所以 (u_\delta) **不是当前式（4）的极小值，也通常不是驻点**。你后续使用

[
-\frac12q^\top D_\delta q
]

构建响应，在数学上可以作为一个 regularized scalar response generator，但它不能继续被描述成“对式（4）的内部坐标进行弛豫”。

这是目前最需要先修正的问题。

### 推荐处理

将数据分为两类：

**稳定结构：**

[
\lambda_{\min}^{opt}>\lambda_{\mathrm{cut}}
]

使用真正的光学子空间逆或者 Cholesky/线性求解，此时确实来自原始 quadratic potential 的弛豫。

**软模或不稳定结构：**

使用 (D_\delta)，但把它命名为：

> signed regularized optical Green's operator

不要称作 equilibrium relaxation。负模结构对应的是 saddle-point linear response，而不是稳定热力学平衡。

论文中可以明确写：

[
\bar G_\delta
=============

G_{\mathrm{direct}}
-\frac12q^\top D_\delta q
]

是单独定义的 regularized response generator；只有在 (\delta\to0) 且非零光学模可逆时，它才恢复真正的 relaxed potential。

另外，当前滤波器

[
f_\delta(\lambda)=
\frac{\lambda}{\lambda^2+\delta^2}
]

在 (|\lambda|=\delta) 附近最大为 (1/(2\delta))，且零点附近对特征值的导数尺度约为 (1/\delta^2)。使用 (\delta=10^{-3}) 时可能造成非常尖锐的梯度，因此必须报告完整的 (\delta) 敏感性，而不能只证明梯度是 finite。

---

# 三、第二个数学问题：当前的 SI response generator 单位不自洽

式（4）看起来是以 eV 为单位的 cell energy：

* (u)：Å；
* (\Phi)：eV/Å²；
* (\Lambda)：eV/Å；
* (Z^*)：电子电荷单位；
* (E)：应当是 V/Å。

但式（6）到式（8）已经把输出转换为：

* (e)：C/m²；
* (C)：GPa；
* (\chi)：无量纲相对 susceptibility。

随后又直接写：

[
\bar\Phi
========

-E^\top e\eta
+\frac12\eta^\top C\eta
-\frac12E^\top\chi E.
]

如果这是物理能量或者能量密度，三个项目前并不具有统一单位。特别是 dielectric 项需要 (\epsilon_0)，cell energy 还需要体积和单位转换。

可以选择两套严格写法之一。

### 写法 A：全部使用内部 cell-energy 单位

网络内部使用：

[
M_e=Z^{*\top}\Phi^{-1}\Lambda
]

等 cell-level coefficients，单位保持为 eV 系统；只有最终计算指标时再除以 (\Omega) 并转换成 SI。

### 写法 B：定义 SI 能量密度

[
g=
-E^\top e\eta
+\frac12\eta^\top C\eta
-\frac12\epsilon_0E^\top\chi E,
]

其中：

* (E)：V/m；
* (C)：Pa；
* (e)：C/m²；
* (g)：J/m³。

cell energy 则为 (G=\Omega g)。

如果不解决，审稿人很容易质疑“thermodynamic potential”只是形式上的生成器，而非物理量。

---

# 四、现在为什么学不动

论文把原因总结为“样本效率和多任务梯度冲突”，但证据还不足。更直接的问题是：

## 1. 你在用 98 个晶体学习极高维的对象

每个晶体需要预测：

[
\Phi\in\mathbb R^{3N\times3N},
\qquad
\Lambda\in\mathbb R^{3N\times6},
\qquad
Z^*\in\mathbb R^{3N\times3}.
]

对于 (N=48)，单个 (\Phi) 就是 (144\times144)。虽然局部 pair head 会共享参数，但监督规模仍然极小。当前真正用于 factor training 的只有 98 个训练晶体，验证集只有 14 个。

最关键的是，printed (\Lambda) 的中位覆盖率只有 25.46%，而当前 (\Lambda) skill 基本为零。压电晶格贡献又恰好线性依赖：

[
Z^{*\top}\Phi^{-1}\Lambda.
]

因此宏观响应学不出来并不意外。

## 2. 没有 mode gauge，不等于因子可辨识

原子坐标表示确实避免了简并声子特征向量的任意旋转，但仅凭总压电张量：

[
e^{ion}=Z^{*\top}\Phi^{-1}\Lambda
]

仍然不能唯一恢复完整的 (Z^*,\Phi,\Lambda)。

总压电标签只有 18 个数，而完整 (\Lambda) 有 (18N) 个分量，(\Phi) 更多。大量不同因子组合可以得到相同宏观响应。

所以当前问题不只是 data scarcity，而是：

> 宏观标签对完整微观因子的 observational identifiability 不足。

只有 98 个带部分直接因子监督的样本，无法消除这种自由度。

## 3. 当前 factor metric 和最终响应不对齐

对 (\Phi) 报普通 component MAE，并不能说明最重要的低频模是否正确。响应中存在 (1/\lambda_m)，低频 polar mode 的一点误差就可能完全改变最终结果，而大量高频 mode 的 component MAE 对响应几乎无关。

必须增加：

* 最低若干 optical eigenvalue error；
* soft-mode sign accuracy；
* soft-mode subspace overlap；
* mode effective charge；
* strain-mode coupling；
* mode-wise piezo contribution；
* response-weighted force-constant error。

---

# 五、最重要的诊断实验

下一步不应先换 optimizer，而应做 **oracle factor replacement**：

| 实验                                        | 用途                   |
| ----------------------------------------- | -------------------- |
| true (Z^*) + true (\Phi) + pred (\Lambda) | 测试 (\Lambda) 是否是主要瓶颈 |
| true (Z^*) + pred (\Phi) + true (\Lambda) | 测试 force constant    |
| pred (Z^*) + true (\Phi) + true (\Lambda) | 测试 BEC               |
| pred (Z^*) + pred (\Phi) + true (\Lambda) | 检查 (Z^*,\Phi) 联合误差   |
| true factors + 当前 (D_\delta)              | 测试 damping 本身的偏差     |
| true factors + exact inverse              | 获得理论上限               |

这是最能解释“factor MAE 改善但 response 没改善”的实验。

还需要按下面两组分别评估：

* 所有光学模为正的 stable structures；
* 含负光学模的 unstable structures。

当前将两者混合后讨论“relaxed response”，物理含义不够干净。

---

# 六、当前 reciprocal idea 仍未和主要物理路径真正结合

这是现在剩余的“拼接感”。

cross-spectrum 最终进入的是：

[
e^{el}=e^{(0)}+e^{spec},
]

即直接电子压电分支。

但是论文的核心故事是：

> 全局极性模式通过 (\Phi^{-1}) 传播并产生晶格介导响应。

目前 (e^{spec}) 并没有直接进入 (Z^*)、(\Phi) 或 (\Lambda)，所以实际上是：

1. 一个 global reciprocal electronic response branch；
2. 一个 local atom-coordinate ionic factorization branch。

这两者仍然是并排的。

更统一的做法是让 reciprocal operator 产生：

* (\Phi) 的长程或低秩部分；
* (\Lambda) 的 collective strain-force field；
* 或者 response-active optical displacement basis。

例如：

[
\Phi
====

\Phi_{\mathrm{short}}
+
B_{\mathrm{spec}}K_{\mathrm{spec}}B_{\mathrm{spec}}^\top.
]

这样“collective polar modes”才真正通过光学 Green's operator 传播。

显式长程电荷对 LO–TO splitting、介电和声子响应的重要性也正成为近期工作的重点；最近模型已经通过环境依赖电荷和长程项恢复 LO–TO splitting。([arXiv][1])

---

# 七、我最推荐的实现升级

## 不再直接从零预测完整 (\Phi) 和 (\Lambda)

改为从一个共享的 strain-aware energy model 求导：

[
\Phi
====

\frac{\partial^2E_\theta}
{\partial u,\partial u},
\qquad
\Lambda
=======

-\frac{\partial^2E_\theta}
{\partial u,\partial\eta}.
]

这样会自动获得：

* (\Phi) 的 Hessian integrability；
* (\Phi=\Phi^\top)；
* (\Phi) 与 (\Lambda) 来自同一个势能面；
* 平移和旋转约束；
* 可以利用大量 energy/force/stress 预训练数据；
* 不再依靠 98 个晶体从零学习整个动力学算子。

目前的 (\Phi=P(H+H^\top)P/2) 只能保证对称和 acoustic sum rule。它是一个 force-constant matrix，但严格来说不是由某个学习能量求导得到的 Hessian，因此跨结构扰动时不保证可积性。

近期已经有统一 electric-enthalpy 模型从同一个标量函数导出 force、polarization、BEC 和 polarizability，也有跨材料的 field-aware 等变模型；因此你的创新不能表述成“第一次从势求响应”，而应强调：

> 在跨化学体系中，通过原子内部坐标的 Schur complement 联合生成 relaxed piezoelectric、dielectric 和 elastic tensors，并显式学习 strain–phonon coupling。

([Nature][2])

物理分解式机器学习也已经用于联合 BEC 和 phonons 计算 ionic dielectric response，因此论文必须补充这一最接近的工作。你的新增价值是 (\Lambda)、压电/弹性耦合、atom-coordinate solve 和晶胞基不变 reciprocal operator。([arXiv][3])

### 实现上不必显式构造稠密 Hessian

只需要求解：

[
\Phi x=q.
]

可以用 energy model 的 Hessian-vector product 和迭代求解：

* stable：CG；
* indefinite：MINRES；
* damped normal equation：
  [
  (\Phi^2+\delta^2P)x=\Phi q.
  ]

这样不需要显式存储 (3N\times3N) 矩阵，才能扩展到 defect supercell 和有限温度快照。当前 dense inverse 的时间约为 (O(N^3))，虽然 48 原子没问题，但不符合论文后面声称的真实世界大超胞方向。

---

# 八、论文现在应如何收缩主贡献

建议正文只保留三个核心贡献：

1. **Atom-coordinate relaxed-response factorization**
   从 (Z^*,\Phi,\Lambda) 的共同算子产生 (e,\chi,C)。

2. **Cell-basis-invariant tensorial reciprocal operator**
   处理 origin、rotation、wrapping 和 (GL(3,\mathbb Z)) 变换，并真正进入 ionic response path。

3. **Quality-gated DFPT factor benchmark and response-aware diagnosis**
   包括真实 factor labels、不稳定模、缺失块 mask 和 oracle factor analysis。

以下内容移到附录：

* random Hessian sketch；
* 通用 masked-species pretraining；
* TRS 的长篇推导；
* 不产生实际收益的 curriculum 细节。

当前 full tensor 只有 18 个分量，Hessian sketch 没有计算优势，继续占据独立方法章节会冲淡主线。

---

# 九、当前版本能不能投

**不能按现在的结果直接作为正常 ICLR 方法论文投稿。**

原因不是 idea 不新，而是：

* 主要指标和 zero predictor 几乎相同；
* 尚未与 GMTNet、EATGNN、CEITNet 做 matched rerun；
* 新架构甚至还没有超过上一版直接张量模型；
* factor test 只有 16 个样本；
* 当前“relaxed potential”有数学和单位表述问题。

但这次方向是对的。优先级应当是：

1. 修正 damped solve 与 potential 的数学关系；
2. 修正 SI/cell-energy 单位；
3. 做 oracle factor replacement；
4. stable/unstable 分开；
5. 扩大 DFPT factor cache；
6. 用共享 energy derivative 产生 (\Phi,\Lambda)；
7. 把 reciprocal operator 接入 ionic pathway；
8. 最后再做多任务 loss balancing。

另外，当前 PDF 还有几处必须立即修复的排版问题：

* 页眉仍是 **ICLR 2026**；
* 摘要出现了 `4.7and0.7predeclared` 的破损句子；
* 多处出现 `Eq. equation 4–equation 8`；
* abstract 中负结果占比过高，尚未形成清晰的正面贡献叙事。

总体上，**idea 已经从“合理组合”升级成了真正值得继续投入的物理学习框架，但当前实现错误地把正则 Green's operator 当成了原势能的弛豫，并且试图用过少因子数据学习过高维、不可充分辨识的微观算子。** 这两个问题解决后，论文才可能真正起飞。

[1]: https://arxiv.org/abs/2603.06396?utm_source=chatgpt.com "Long-range machine-learning potentials with environment-dependent charges enable predicting LO-TO splitting and dielectric constants"
[2]: https://www.nature.com/articles/s41467-025-59304-1?utm_source=chatgpt.com "Unified differentiable learning of electric response"
[3]: https://arxiv.org/abs/2509.26022?utm_source=chatgpt.com "Accelerated Discovery of High-\k{appa} Oxides with Physics-Based Factorized Machine Learning"
