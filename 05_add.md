## 总体判断

**当前版本有一个值得继续做的核心：用周期晶体中的全局极化关联来辅助完整压电张量预测。**
但以 ICLR 2027 的标准看，现阶段还不能称为“突破性进展”，更接近：

> 将若干合理的等变建模、倒空间建模和物理约束技术，围绕压电张量问题进行了一次较有物理动机的整合。

问题不在于方法一定是“抄”或“拼接”，而在于草稿同时声称了局部 Cartesian encoder、极化谱、晶格多帧、响应势、预训练、幅值分解、Hessian sketch 等太多贡献，导致真正独特的部分反而不突出。当前唯一有机会形成独立贡献的，是 **collective polar spectrum**；但它目前仍有物理覆盖不完整和晶胞表示不变性的问题。

# 一、和最新工作的关系

## 1. 局部 Cartesian environment 不是核心新贡献

你的式（8）通过多通道向量、无迹二阶张量和 channel-space mixing 构建高阶输出，而且正文已经承认它沿用了 CEITNet 的 Cartesian many-body 原理。

2026 年的 CEITNet 已经明确提出：

* multi-channel Cartesian local environment；
* channel-space interaction；
* 不依赖高阶 Clebsch–Gordan product；
* 面向 dielectric、piezoelectric、elastic 等二到四阶张量。

这和 PiezoJet 的局部编码器重合度很高。([arXiv][1])

**建议：不要把局部 Cartesian encoder 写成主要创新。**
将其定位成高效 backbone，真正贡献必须放在“从局部响应模式到晶体级响应算子”的机制上。

---

## 2. Polar decomposition 与多帧平均也已有较近先例

GMTNet 已在 ICML 2024 系统处理 O(3) 张量等变性和晶体空间群约束。([Proceedings of Machine Learning Research][2])

GoeCTP 则已经使用 polar decomposition 将晶体变换到标准姿态，并把它作为通用张量预测框架；其后续版本也继续采用 frame averaging 做晶体张量预测。([arXiv][3])

因此，你的六个晶格行排列加 polar factors：

[
Q_m=\operatorname{polar}(P_mL)
]

有一定工程变化，但目前很难被认为是单独的主要理论突破。

**更严重的是，它使方法看起来像：**

> CEITNet local encoder + reciprocal spectrum + GoeCTP frames + GMTNet symmetry projection。

这正是你不希望出现的“拼接感”。

---

## 3. 当前的“response potential”实质上仍是直接张量回归

现在的实现流程是：

1. 网络先预测 (e_\theta(x))；
2. 再定义
   [
   \Phi_\theta(x,\eta,E)=-E^\top e_\theta(x)\eta;
   ]
3. 对 (\Phi) 求导，重新得到 (e_\theta(x))。



这在数学上是正确的，但它没有为 (e) 增加新的物理约束。混合偏导相等也是该双线性表达式的直接结果。因此审稿人很可能会说：

> The potential formulation is a reparameterization of direct tensor regression rather than a learned thermodynamic potential.

相比之下，2025 年 Nature Communications 的工作将电场作为网络输入，真正学习随原子位置和电场变化的 electric enthalpy，再从同一个函数导出力、极化、Born effective charge 和 polarizability，并用于外场分子动力学。([Nature][4])

所以目前不能把“thermodynamic potential”作为强创新。要么降调为 **coefficient interface**，要么真正学习 (\Phi(x,\eta,E)) 的场和应变依赖。

---

## 4. 倒空间极化谱是最有潜力的部分，但不是全新的范式

Latent Ewald Summation 已经使用局部环境预测 latent variables，再通过倒空间/Ewald 求和获得长程信息；后续工作甚至可以从长程 MLIP 中导出极化和 Born effective charge。([Nature][5])

因此，“learned local variables + reciprocal-space structure factor”本身已经存在。

你的区别在于：

> latent variable 是一个极性向量 (p_i)，并专门用于构建三阶压电响应。

这可以成为新意，但你必须证明：

1. 极性向量比 generic scalar latent charge 更适合张量响应；
2. 它确实捕获了局部响应的相干和抵消；
3. 提升主要发生在需要长程协同的材料，而不是普通数据拟合收益；
4. 它能迁移到其他响应，而不仅是 JARVIS 压电标签。

否则审稿人会把它看作 Latent Ewald 的任务特化版本。

---

# 二、当前存在的三个关键科学问题

## 问题一：压电性不等于静态极化相干

草稿的核心表述是：

> piezoelectricity is the crystal-scale coherence or cancellation of local polar distortions.

这个说法对铁电体、极性晶体和软模主导体系很有启发性，但不能覆盖一般压电材料。

20 个压电点群中，只有 10 个是 polar point groups，另外 10 个是**非极性但压电**的点群；这些材料在无应变时没有自发极化，却会在机械应变下产生极化。石英就是典型例子。([TU Graz][6])

因此：

* 静态局部极化的相干不是压电性的必要条件；
* 静态极化较强也不必然意味着压电系数大；
* 压电系数更本质地描述的是**极化对机械应变的导数**。

你的论文自己也写出了更正确的物理分解：

[
e^{\mathrm{ion}}\propto \Omega^{-1}Z^*\Phi^{-1}\Lambda,
]

其中包含 Born effective charge、力常数逆矩阵和 strain–force coupling。

所以，“collective polar coherence”更适合被描述成：

> 与压电响应相关的一种结构先验，而不是压电性的普遍微观机制。

至少必须将测试集按以下类别分别报告：

* polar point groups；
* nonpolar piezoelectric point groups；
* soft-mode/high-BEC materials；
* ordinary low-response materials。

如果方法只在极性晶体上有效，核心论断就需要收缩。

---

## 问题二：固定 reciprocal index set 可能不具有等价晶胞基不变性

你固定使用：

[
H={-1,0,1}^3\setminus{0},
\qquad
g_h=2\pi hL^{-T}.
]

并只对晶格三行的六种排列做平均。

但是同一个 Bravais lattice 可以由不同的 primitive basis 表示：

[
L'=UL,\qquad U\in GL(3,\mathbb Z).
]

其中不仅包括行排列，还包括符号变化和整数剪切。固定的小立方索引集合 (H) 在一般 (U) 下并不封闭，所以同一晶体换一个等价 primitive cell，模型可能得到不同的倒空间特征。

当前 frame ensemble 只处理了六个排列，不能处理完整的晶格基变换。

晶体模型不仅需要旋转、平移和原子置换不变性，还需要处理同一无限周期结构的不同 unit-cell 表示；ICLR 2025 的晶体 frame 工作也明确把 unit-cell variation 作为晶体建模中的额外不变性问题。([OpenReview][7])

### 更正确的实现

不要按固定整数索引截断，而应按**实际 reciprocal vector 的物理长度**截断：

[
\mathcal G(L)=
\left{
g\in\Lambda^*:
0<|g|\le g_{\max}
\right}.
]

然后对整个物理 reciprocal lattice shell 求和。这样更换晶格基只会重新编号同一组 (g)，而不会改变特征。

投稿前必须增加以下等价晶胞审计：

* primitive ↔ conventional cell；
* 随机 (GL(3,\mathbb Z)) unimodular transforms；
* supercell 表示；
* lattice row permutation/sign flip；
* origin shift；
* atom wrapping 和 boundary shift。

这是当前最容易被严谨审稿人抓住的数学漏洞。

---

## 问题三：(|P_h|^2) 丢掉了最重要的方向信息

当前构造：

[
P_h=\frac1N\sum_i p_i e^{2\pi ih^\top f_i},
\qquad
c_s\sim\sum_h w_s(|g_h|)|P_h|^2.
]

它具有 origin invariance，但同时：

* 丢掉 complex phase；
* 丢掉 (P_h) 的方向；
* 最终只作为 invariant scalar context (z)；
* 因而主要只能调制幅值或通道权重；
* 张量方向仍主要来自局部直接预测 (e^{(0)}) 和 lattice frames。

这削弱了“全局极化相干直接决定张量形状”的论断。

### 一个更有原创性的改造

你已经有 chemical structure factor (S_h)，可以定义：

[
u_h=\operatorname{Re}\left[P_hS_h^*\right].
]

在 origin shift 下，(P_h) 和 (S_h) 获得相同相位，因此乘积相位抵消；同时 (u_h) 仍然是一个 O(3)-equivariant polar vector。

随后直接构造三阶响应基：

[
e^{\mathrm{spec}}
=================

\sum_{g\in\mathcal G(L)}
w(g),
\operatorname{sym}_{jk}
\left[
u_g\otimes \hat g\otimes\hat g
\right].
]

这个构造同时具备：

* origin invariance；
* O(3) rank-3 polar equivariance；
* strain-index symmetry；
* 保留全局极化方向；
* 不依赖 polar canonical frame；
* 若使用物理 reciprocal shells，还可实现晶格基不变性。

这会比“scalar spectrum 条件化一个多帧 MLP”更统一，也更不像现有工作的拼接。

还应显式区分：

* (g=0)：uniform ferroelectric/polar mode；
* (g\neq0)：antiferroelectric、modulated 或局部抵消模式。

---

# 三、我最推荐的核心重构：从“极化特征”迁移到“响应算子”

真正能将论文提升一个层级的，不是继续增加网络模块，而是把任务从：

> 从静态晶体直接回归 (e)

改为：

> 从静态晶体学习产生 (e) 的、可分解且可迁移的晶格响应算子。

## Mode-resolved response factorization

在正常模基底下，可以示意性写成：

[
e_{\alpha\mu}
=============

e^{\mathrm{el}}*{\alpha\mu}
+
\frac1\Omega
\sum_m
\frac{
Z^{*}*{\alpha m}\Lambda_{m\mu}
}{
\omega_m^2+\epsilon
},
]

其中：

* (e^{\mathrm{el}})：clamped-ion electronic response；
* (Z^{*}_{\alpha m})：第 (m) 个模式的 mode effective charge；
* (\omega_m^2)：模式刚度；
* (\Lambda_{m\mu})：应变与模式的耦合；
* 低频软模和异常 Born charge 自然产生大压电响应。

这和你的“collective polar motif”具有连续关系，但发生了实质迁移：

> 从学习静态极化模式，变成学习极化模式对外部应变的响应传播过程。

这更符合真实物理，也自然解释为什么少数材料具有异常大的压电系数。

更关键的是，JARVIS 的 DFPT 数据不只有总压电张量。公开资料显示，其约 5,015 个材料还计算了 Γ 点 phonons、Born effective charges、piezoelectric 和 dielectric tensors；数据库文档还明确包含 phonon eigenvectors 以及 dielectric 的 ionic/electronic components。([Nature][8])

因此你可以：

1. 用 BEC 和 phonon eigenmode/frequency 监督 (Z^*_m) 与 (\omega_m)；
2. 将 (\Lambda_m) 作为受总压电标签约束的 latent strain-mode coupling；
3. 用一个较小的附加 DFPT 子集监督 (\Lambda_m)；
4. 把剩余误差解释为 electronic/clamped-ion branch；
5. 从同一组模式同时预测 dielectric、IR intensity 和 piezoelectric response。

这样论文的贡献会从“一个新 GNN”变成：

> 一个可辨识、模式分辨、可跨响应迁移的晶体电机耦合学习框架。

这更接近 ICLR 喜欢的“新学习问题与新归纳偏置”，而不仅是材料任务上的模型改造。

---

# 四、真正的统一势应怎样实现

当前不要先预测 (e) 再把它放回 (\Phi)。更有意义的方案是引入内部模坐标 (q)：

[
\Phi_\theta(x,q,\eta,E)
=======================

\Phi_0(x)
+\frac12 q^\top K_\theta q
-q^\top\Lambda_\theta\eta
-E^\top Z^**\theta q
-E^\top e^{\mathrm{el}}*\theta\eta
+\frac12\eta^\top C_\theta\eta
-\frac12E^\top\chi^{\mathrm{el}}_\theta E.
]

消去或优化内部坐标：

[
q^*=K_\theta^{-1}
\left(\Lambda_\theta\eta+Z_\theta^{*\top}E\right),
]

再从 relaxed potential 的导数得到压电、介电和弹性响应。

这样：

* (K^{-1}) 显式表示软模放大；
* (Z^*) 表示电场–模式耦合；
* (\Lambda) 表示应变–模式耦合；
* (e,C,\chi) 不再是三个独立 head；
* Maxwell relations 和响应间耦合不再只是形式上的；
* 可以对 (K)、(C) 和整体 Hessian 加稳定性约束。

这才是真正有可能称作 **response potential learning** 的方案。

---

# 五、怎样更符合真实世界

## 1. 不要完全依赖 hard point-group projection

当前通过 Reynolds projection 保证 centrosymmetric 样本严格输出零，这在理想 DFT 晶体上很干净，但真实材料包含：

* 缺陷；
* 掺杂；
* 局部无序；
* 有限温度振动；
* 应力；
* 实验结构误差；
* 名义高对称但局部破缺的结构。

硬投影会把这些真实的小响应直接抹掉。当前“point-group residual 为零”主要证明后处理投影正确，而不是网络真正学会了空间群规律。

建议改成：

[
e(x)
====

P_G[e_{\mathrm{sym}}(x)]
+
g(\delta_G(x))e_{\mathrm{break}}(x),
]

其中 (\delta_G(x)) 表示结构偏离理想空间群轨道的程度；在理想晶体上 (g(0)=0)，存在缺陷或热扰动时允许连续出现 symmetry-breaking response。

---

## 2. 加入结构扰动和有限温度评估

至少构造：

* 对称保持的小位移；
* 对称破缺的小位移；
* 体应变和剪切应变；
* vacancy/substitution 小规模超胞；
* phonon-mode distortions；
* 多个有限温度快照。

重点测试：

[
e(x+\delta x)-e(x)
]

是否具有合理的连续性，而不只是 relaxed ideal structure 上的静态 MAE。

---

# 六、哪些内容应保留，哪些应该降级

| 当前模块                                  | 建议                                                 |
| ------------------------------------- | -------------------------------------------------- |
| Collective polar mechanism            | **保留并重构为 response operator**                       |
| Reciprocal-space construction         | **保留，但改成物理 reciprocal-shell + tensorial spectrum** |
| Cartesian local encoder               | 保留作 backbone，不声称核心创新                               |
| Six polar lattice frames              | 优先删除；若保留，仅作 baseline/ablation                      |
| Bilinear coefficient potential        | 替换为 mode-conditioned genuine response potential    |
| Activation–amplitude–shape            | 可以保留为优化技巧，但不列主贡献                                   |
| Masked species + coordinate denoising | 移到 appendix，或换成 mode/strain-aware pretraining      |
| Random Hessian sketches               | 移到 appendix；当前 18 维输出中没有速度或显存优势                    |
| TRS 新指标                               | 可作为辅助指标，不能替代标准指标                                   |
| Reynolds hard projection              | 理想晶体 baseline；主模型增加 soft symmetry-breaking branch  |

尤其是 Hessian sketch：草稿自己的实验已经显示 explicit full loss 更快且不更耗显存，因此把它列为 contribution 会分散注意力。

# 最终建议

**不要继续在当前框架上再加模块。**

最好的主线是：

> 从“collective polar feature”升级为“symmetry-resolved collective response operator”，显式学习 Born charge、soft-mode stiffness 和 strain-mode coupling，并从一个真正的响应势中联合导出压电、介电与弹性响应。

推荐将论文的三个主要贡献压缩成：

1. **Mode-resolved electromechanical response factorization**：将稀有大响应解释为 charge–softness–strain coupling，而不是黑盒幅值回归。
2. **Cell-basis-invariant tensorial reciprocal response operator**：保留全局极化方向，同时满足 origin、O(3) 和等价晶胞不变性。
3. **Transfer under response type and symmetry breaking**：从 piezoelectric 迁移到 dielectric/BEC/IR，并在扰动、缺陷或有限温度结构上验证。

这会形成清晰的研究迁移：

[
\text{直接张量预测}
\quad\longrightarrow\quad
\text{产生张量的物理响应算子学习},
]

而不是：

[
\text{CEITNet}
+\text{reciprocal feature}
+\text{frame averaging}
+\text{potential wrapper}
+\text{pretraining}.
]

按当前版本直接投 ICLR 2027，创新性和证据都偏弱；按上述路线重构后，它才有机会从“合理的材料模型”变成“具有可迁移物理机制的新学习范式”。另外，草稿页眉仍写的是 **ICLR 2026**，后续需要统一改成 ICLR 2027。

[1]: https://arxiv.org/abs/2602.04323?utm_source=chatgpt.com "Efficient Equivariant High-Order Crystal Tensor Prediction via Cartesian Local-Environment Many-Body Coupling"
[2]: https://proceedings.mlr.press/v235/?utm_source=chatgpt.com "Volume 235: International Conference on Machine Learning ..."
[3]: https://arxiv.org/abs/2410.02372?utm_source=chatgpt.com "Fast Crystal Tensor Property Prediction: A General O(3)-Equivariant Framework Based on Polar Decomposition"
[4]: https://www.nature.com/articles/s41467-025-59304-1?utm_source=chatgpt.com "Unified differentiable learning of electric response"
[5]: https://www.nature.com/articles/s41524-025-01577-7?utm_source=chatgpt.com "Latent Ewald summation for machine learning of long- ..."
[6]: https://lampz.tugraz.at/~hadley/ss2/crystalphysics/piezo.php?utm_source=chatgpt.com "Piezoelectricity"
[7]: https://openreview.net/pdf?id=gzxDjnvBDa&utm_source=chatgpt.com "RETHINKING THE ROLE OF FRAMES FOR SE(3)"
[8]: https://www.nature.com/articles/s41524-020-0337-2?utm_source=chatgpt.com "High-throughput density functional perturbation theory and ..."
