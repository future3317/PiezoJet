# 总评

以你本条给出的最新三 seed 结果为准。我的审稿结论会是 **major revision，但不是否定因子化方向**。当前证据最支持以下三点：

1. **现有 detached Moore–Penrose chart 在 (A_\theta) 满行秩时，把所谓 factorized ionic forward 代数上退化成了 direct ionic head。** 因而当前的 factorized–direct comparison 并没有真正比较“因子化物理预测”和“直接预测”。
2. **“4,961 条 total labels 没有解决 ionic collapse”目前只是 fixed-budget 结论，不是数据规模结论。** 100 joint optimizer updates 很可能连一个 macro epoch 都没有覆盖。
3. 排除上述两点后，主要科学瓶颈仍然很清楚：**response-active internal displacement/strain coupling，即 (U_\eta=\mathcal D(\Phi)\Lambda) 或其低维响应子空间，跨材料泛化失败。** 你已有的 true-(Z^*,\Phi)+predicted-(\Lambda) oracle 几乎零方向相关和约 5% 振幅，已经相当直接地指向这一点。

我会把问题分为：

| 类别               | 结论                                                                                                                                    |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **必须修正的数学/实现错误** | detached chart 的恒等退化与 ghost gradient；full-corpus exposure 定义；光学基 (Q) 的公式歧义；Voigt/Kelvin 范数与逐分量 robust loss 的旋转不变性；branch target 投影一致性 |
| **可验证的架构假设**     | signed resolvent 的软模 shortcut；直接预测 (U_\eta)；质量加权与 scale–shape 参数化；低秩 response subspace；physical/macro 双塔                              |
| **后续研究方向**       | LO–TO/NAC 明确边界条件；SSCHA/TDEP 有限温自由能 Hessian；联合概率潜变量与响应导向主动采样                                                                           |

---

# 1. 为什么 direct total 可能略有信号，而 factorized ionic 几乎完全塌缩？

## 1.1 当前 direct total 的“略有信号”统计上仍等同于零

你的 matched direct total TRS 为

[
0.00444\pm0.01999,
]

其 seed 波动远大于均值。严格地说，这不是已建立的正信号，只能说它没有显著劣于零预测。factorized 与 direct 的 paired difference 也同样不显著。

即便 direct total 最终能学到一点，它也比因子化问题容易得多：

* 输出只有 18 个宏观可观测分量；
* 4,961 个样本都提供监督；
* 空间群大幅约束允许的张量形状；
* 模型可以学习“零/近零—非零”、晶体系统模板和条件均值；
* 它不必恢复原子位移的相位、软模方向或电子/离子分支划分。

GMTNet 和 EATGNN 已经说明直接等变预测完整压电张量是一个可行的问题类别，但它们的拆分、数据清洗和指标与 PiezoJet 不同，必须在同一 formula-disjoint split 上重跑才能作数。([Proceedings of Machine Learning Research][1])

## 1.2 当前 detached chart 在满行秩时是严格退化的

令

[
y_\theta=\frac{\Omega}{c_e}\hat e_\theta^{ion},
\qquad
\bar A_\theta=\operatorname{sg}(A_\theta),
]

则你当前定义

[
\hat\Lambda=
\bar A_\theta^+y_\theta+
(I-\bar A_\theta^+\bar A_\theta)R_{\rm null}.
]

虽然 autograd 图中 (A_\theta) 与 (\bar A_\theta) 不同，但在数值 forward 中二者完全相等。若 (A_\theta\in\mathbb R^{3\times(3N-3)}) 满行秩，则

[
\begin{aligned}
A_\theta\hat\Lambda
&=
A_\theta\bar A_\theta^+y_\theta+
A_\theta(I-\bar A_\theta^+\bar A_\theta)R_{\rm null}\
&=y_\theta.
\end{aligned}
]

因此

[
\boxed{
\frac{c_e}{\Omega}A_\theta\hat\Lambda
=

\hat e_\theta^{ion}
}
]

在浮点和 SVD cutoff 误差之外恒成立。当前稿件所描述的 chart 和单独 consistency loss 正是这一结构。

由此有三个严重后果。

### 后果 A：consistency loss 基本恒为零

[
L_{\rm cons}
=

\rho!\left(
\frac{c_e}{\Omega}A_\theta\hat\Lambda-\hat e_\theta^{ion}
\right)
]

在满行秩区间前向残差为零。SmoothL1、L2 等在零残差处的梯度也为零，所以它不能“校准” (Z^*,\Phi,\Lambda)。只有 predicted (A) 数值降秩或 cutoff 生效时才产生残差，而那恰恰是最不可靠的区域。

你报告的 true (A) 有 96.1% 满行秩并不能解决此问题；真正应记录的是**每一步 predicted (A_\theta) 的三个奇异值和 cutoff 后秩**。对随机或一般神经网络输出的 (3\times d) 矩阵，满行秩通常反而是常态。

### 后果 B：所谓 factorized ionic prediction 实际就是 direct ionic prediction

所以当前的 factorized ionic amplitude、cosine 与 direct ionic head 不是两个独立模型预测。它们只是在同一个 direct head 上，附加了一条人为的反向传播路径。

这意味着目前不能用 factorized–direct paired metric 判断 factorization 是否有优势。

### 后果 C：attached (A) 产生 ghost/straight-through gradient

autograd 把 (\hat\Lambda) 看作不依赖 (A_\theta)，于是计算

[
d(A_\theta\hat\Lambda)=dA_\theta,\hat\Lambda.
]

但如果每次 forward 都重新计算 chart，真正的满秩函数值始终为 (y_\theta)，其关于 (A_\theta) 的总导数应该为零。也就是说：

* 有限差分：改变 (A)，重算 pseudoinverse 后，宏观输出基本不变；
* 当前 autograd：给出非零 (A)-梯度。

这不是某个良定义 factorized model 的梯度，而是 chart-dependent straight-through estimator。它可能持续改变 (Z^*,\Phi)，但下一次 forward 又被新的 pseudoinverse 精确补偿，因此宏观损失不改善而 factor heads 漂移。

这是我认为当前优先级最高的**必须修正错误**。

## 1.3 full-(\Lambda) loss 与 direct ionic loss 在错误 (A) 下也会冲突

对 strict 样本，full-(\Lambda) loss 希望

[
\bar A_\theta^+y_\theta+
(I-\bar A_\theta^+\bar A_\theta)R_\theta
\approx \Lambda^\star .
]

由于 (\bar A_\theta) detached，该损失不能修正错误的 (A_\theta)，只能迫使

[
y_\theta\approx \bar A_\theta\Lambda^\star.
]

但 direct ionic 标签要求

[
y_\theta\approx A^\star\Lambda^\star.
]

当 (\bar A_\theta\neq A^\star) 时，两个监督直接争夺同一个 (y_\theta)。这不是普通 loss 权重问题，而是 chart 参数化本身构造出的冲突。

## 1.4 total labels 无法识别电子/离子分支

在只有 total 标签时，

[
e^{tot}=e^{el}+e^{ion}
]

允许任意

[
e^{el}\rightarrow e^{el}+\Delta,
\qquad
e^{ion}\rightarrow e^{ion}-\Delta.
]

因此多出的 4,351 条左右 total-only 样本不会自动告诉模型 ionic amplitude 应该是多少。若 electronic/direct macro head 更容易优化，最简单的解就是：

[
\hat e^{ion}\rightarrow 0,\qquad
\hat e^{el}\rightarrow\hat e^{tot}.
]

这正是 branch-allocation non-identifiability，而不是简单的多任务梯度冲突。

## 1.5 逐因子 point estimate 的乘积天然有 attenuation bias

即使没有 chart 问题，由独立或近独立的 deterministic heads 给出条件均值，也有

[
\mathbb E[
Z^{*T}\mathcal D(\Phi)\Lambda\mid x]
\neq
\mathbb E[Z^*\mid x]^T
\mathcal D(\mathbb E[\Phi\mid x])
\mathbb E[\Lambda\mid x].
]

右侧丢失了：

* (Z^*) 与软模方向的相关性；
* (\Lambda) 与软模方向的相关性；
* 模态符号和跨模 cancellation；
* 非线性逆算子带来的 Jensen bias。

尤其当 strain-active mode 在相近结构之间改变符号或模式混合时，逐分量条件均值会趋近零，而真实响应的条件均值未必为零。这是 factorized amplitude collapse 的一般机制。

---

# 2. signed regularized resolvent 是否可能是主因？

## 2.1 它确实具有错误的软模单调性

定义

[
f_\delta(\lambda)=\frac{\lambda}{\lambda^2+\delta^2}.
]

相对于静态逆 (1/\lambda)，保留比例为

[
\frac{f_\delta(\lambda)}{1/\lambda}
=

\frac{\lambda^2}{\lambda^2+\delta^2}.
]

因此：

* (|\lambda|=10\delta)：保留 99.0%；
* (|\lambda|=3\delta)：保留 90%；
* (|\lambda|=\delta)：保留 50%；
* (|\lambda|=0.1\delta)：只保留 0.99%；
* (\lambda\rightarrow0)：响应趋于零。

而物理 harmonic static response 在稳定侧应随 (\lambda\to0^+) 增大，而不是消失。Takigawa 等人的 factorized ionic dielectric 工作也明确显示静态模态贡献随 (1/\omega_m^2) 增强，并只在动态稳定材料上使用静态逆。([arXiv][2])

同时

[
f_\delta'(\lambda)
=

\frac{\delta^2-\lambda^2}
{(\lambda^2+\delta^2)^2},
\qquad
f_\delta'(0)=\frac{1}{\delta^2}.
]

对 (\delta=10^{-3})，零点附近的谱导数尺度为 (10^6)。所以“输出有界”并不意味着“反向传播条件良好”。

## 2.2 它不是静态平衡 Green 函数，而是 Tikhonov normal-equation 解

在光学子空间中，

[
\mathcal D_\delta(\Phi)
=

\Phi(\Phi^2+\delta^2I)^{-1}.
]

因此

[
u_\delta=\mathcal D_\delta(\Phi)\Lambda
]

是以下问题的解：

[
u_\delta
=

\arg\min_u
\left[
|\Phi u-\Lambda|_2^2+
\delta^2|u|_2^2
\right].
]

它最小化的是**力平衡残差加位移惩罚**，不是

[
\Phi u=\Lambda
]

的静态平衡解。它可以作为一个明确的 regularized estimator，但不应与不稳定参考结构的静态 equilibrium response 混称。

另外，任何同时满足“连续、奇函数、符号保持、在零附近有界”的谱滤波 (f) 都必有 (f(0)=0)。所以不存在一个魔法型连续 signed filter，既能穿过零点保持有界，又能保留静态软模发散。这里必须改变**物理语义**，而不是继续寻找另一条平滑曲线。

## 2.3 但从你已有证据看，它不像主因

你已经观察到：

* true stable factors 上 exact 与 regularized propagation 基本一致；
* true (Z^*,\Phi)+predicted (\Lambda) 仍只有约 5% 振幅且 cosine 约零；
* response-weighted (\Phi) error 相对小。

因此更合理的排序是：

[
\boxed{
\text{chart 退化 / exposure}

>

\text{response-active }\Lambda\text{ 或 }U_\eta

>

\text{predicted soft spectrum}

>

\text{固定 }\delta\text{ 本身}
}
]

但 predicted (\Phi_\theta) 仍可能把很多特征值推到 (|\lambda|\ll\delta)，利用“越软反而越接近零响应”的 shortcut。必须直接审计，而不能只看 true-factor spectrum。

## 2.4 我建议的连续、可微替代

### 方案 A：把 raw signed Hessian 与 static-response Hessian 分开

保留

[
\Phi_{\rm raw,\theta}
]

作为 DFPT signed force-constant 预测和 instability diagnostic；另设始终正定的

[
K_{\rm stat,\theta}
=

L_\theta L_\theta^T+
\epsilon s_K I
]

或由 PSD bond/local-star energy 构造，用于静态响应：

[
U_\eta=(K_{\rm stat,\theta})^{-1}\Lambda_\theta.
]

这不需要任何按 predicted (\lambda_{\min}) 的 hard switch。两个 head 有不同语义：

* (\Phi_{\rm raw})：零温参考构型的 signed curvature；
* (K_{\rm stat})：可定义静态响应的有效正定 curvature。

在 true-stable 样本上监督二者一致；对 unstable 样本，如果没有有限温或稳定相标签，不应声称 (K_{\rm stat}) 是 raw DFPT Hessian。

若仍需有界，可使用单调的 positive ridge：

[
U_\eta=(K_{\rm stat}+\epsilon s_KI)^{-1}\Lambda,
]

其谱滤波 (1/(\kappa+\epsilon s_K)) 在 (\kappa\to0^+) 达到上限，而不是回落到零。

### 方案 B：使用尺度协变的 relative regularization

当前绝对 (\delta=10^{-3}) 对不同刚度尺度的材料含义不同。定义

[
s_K=
\sqrt{\frac{1}{d}\operatorname{tr}K^2},
\qquad
\bar K=K/s_K,
]

然后

[
\mathcal D_\epsilon(K)
=

\frac{1}{s_K}
\bar K(\bar K^2+\epsilon^2I)^{-1}.
]

这样 (\epsilon) 是无量纲相对分辨率。它仍然只是 regularized continuation，但不会把材料间刚度尺度差异混进固定 (\delta) 偏差。

### 方案 C：直接预测 (U_\eta)，用方程残差监督，而不是对 inverse 反传

定义

[
U_\eta=\mathcal D(\Phi)\Lambda.
]

对稳定静态模型，施加

[
L_{\rm eq}
=

|K_{\rm stat}U_\eta-\Lambda|^2.
]

若暂时保留当前 Tikhonov 语义，则其精确 normal equation 是

[
(\Phi^2+\delta^2I)U_\eta=\Phi\Lambda,
]

所以可训练

[
L_{\rm normal}
=

|(\Phi^2+\delta^2I)U_\eta-\Phi\Lambda|^2.
]

这样 inverse 不再是主要梯度通道。

### 方案 D：对 unstable 材料使用自由能 Hessian，而不是 signed static continuation

真正的有限温动态稳定应来自自由能曲率，例如 SSCHA 或 TDEP 有效 force constants，而不是对零温 saddle Hessian 施加任意谱滤波。SSCHA 明确以量子/热自由能 Hessian 描述有限温稳定性；TDEP 从有限温系综拟合有效 harmonic force constants。([arXiv][3])

---

# 3. (Z^*,\Phi,\Lambda) 是否有尺度、gauge 和条件数不可辨识性？

答案是肯定的，而且“(A) 满行秩”并不减少主要不可辨识性。

## 3.1 response-null gauge

光学维数为

[
d=3N-3.
]

对满行秩 (A\in\mathbb R^{3\times d})，每个 strain column 的 null dimension 为

[
d-3=3N-6.
]

六个 strain columns 总共仍有

[
6(3N-6)
]

个宏观压电完全看不见的自由度。满行秩只说明每个三维 polarization response 都可实现，绝不说明 (\Lambda) 可识别。

正确的 strict-label 分解应基于**真因子**：

[
P_{\rm act}^\star=A^{\star+}A^\star,
]

[
\Lambda_{\rm act}^\star
=P_{\rm act}^\star\Lambda^\star,
\qquad
\Lambda_{\rm null}^\star
=(I-P_{\rm act}^\star)\Lambda^\star.
]

然后让一个独立的 (\Lambda_\theta) 或 (U_{\eta,\theta}) head 分别接受 active/null 监督。不要用当前 predicted (A_\theta) 自己定义 chart 后再声称它识别了 active 部分。

## 3.2 branch-allocation gauge

total-only 样本存在

[
e^{el}\leftrightarrow e^{ion}
]

的任意重新分配。这只能通过 branch labels、结构化先验或概率模型解决，不能由 total 数量本身解决。

## 3.3 exact factorization 的尺度 gauge

对 exact inverse，变换

[
Z^*\to aZ^*,
\qquad
\Phi\to c\Phi,
\qquad
\Lambda\to b\Lambda,
]

只要

[
\frac{ab}{c}=1,
]

宏观响应不变。原子级 labels 会在 149/610 样本上打破该 gauge，但在 total-only 样本上不会。

固定 (\delta) 的 resolvent 会破坏这个精确 gauge，但它是用一个任意绝对尺度引入偏差，而不是获得物理辨识性。

## 3.4 模态 gauge

单个 eigenvector 存在符号 gauge；简并子空间存在任意 (O(r)) 旋转 gauge。逐 mode eigenvector matching 在近简并区间没有稳定含义。应监督 projector、子空间主角度或 cluster response contribution，而不是逐 eigenvector。

## 3.5 推荐的预条件与重参数化

### 质量加权

[
K=M^{-1/2}\Phi M^{-1/2},
\quad
\widetilde Z=M^{-1/2}Z^*,
\quad
\widetilde\Lambda=M^{-1/2}\Lambda,
]

则

[
e^{ion}
=

\frac{c_e}{\Omega}
\widetilde Z^TK^{-1}\widetilde\Lambda.
]

这把 phonon stiffness、mode normalization 和原子质量放到标准 dynamical-matrix 几何中。平移投影必须同时改成质量加权平移向量

[
\tau_{\kappa a,b}
=

\frac{\sqrt{m_\kappa}\delta_{ab}}
{\sqrt{\sum_\kappa m_\kappa}}.
]

### scale–shape 分解

[
\widetilde Z=s_Z\bar Z,
\qquad
K=s_K\bar K,
\qquad
\widetilde\Lambda=s_\Lambda\bar\Lambda,
]

于是

[
e^{ion}
=

\frac{c_e}{\Omega}
\frac{s_Zs_\Lambda}{s_K}
\bar Z^T\bar K^{-1}\bar\Lambda.
]

分别预测：

* (\log s_Z,\log s_K,\log s_\Lambda)；
* unit-RMS 的 equivariant shapes；
* 宏观 amplitude scale
  [
  a_{\rm ion}
  ===========

  \log\frac{c_es_Zs_\Lambda}{\Omega s_K}.
  ]

这样不会要求网络通过三个未归一化 heads 的偶然乘积生成正确幅度。

### irrep whitening

在不破坏等变性的前提下，可对每个 (l^\pi) block 及其 multiplicity space 分别 whitening。不能对 Cartesian components 任意独立标准化。

### 联合概率潜变量，而非三个独立点估计

令

[
h\sim q_\theta(h\mid x),
\qquad
(Z,K,U_\eta)=g_\theta(x,h),
]

训练时最小化已观测标签的 marginal likelihood，并用 Monte Carlo 估计

[
\mathbb E_h[
Z(h)^TU_\eta(h)].
]

这保留因子间相关性；否则“posterior mean 的乘积”会系统性丢失 covariance。初始实现可用低秩 Gaussian latent + Student-(t) observation noise，不必一开始使用复杂 flow。

---

# 4. 应否直接预测可观测中间量？

**是。我的第一选择不是 mode effective charge 本身，而是直接预测内部位移响应**

[
\boxed{
U_\eta=\frac{\partial u}{\partial\eta}
=\mathcal D(\Phi)\Lambda
}
]

然后

[
e^{ion}
=

\frac{c_e}{\Omega}Z^{*T}U_\eta.
]

它同时具备以下优点：

* 避免最病态的 inverse–product 路径；
* 比完整 (\Phi) 少 (O(N)) 到 (O(N^2)) 输出；
* 比完整 (\Lambda) 更接近真正决定压电响应的量；
* 可由有限应变 relaxation 直接验证；
* 仍保留原子位移和模式解释。

## 4.1 精确的等变输出形式

每个原子输出

[
U_{\kappa,a,jk}=U_{\kappa,a,kj},
]

其中 (a) 是位移方向，(jk) 是对称 strain tensor。它是 odd-parity、末两指标对称的三阶 Cartesian tensor，分解为

[
2\times 1_o\oplus 2_o\oplus 3_o.
]

在 e3nn 中可直接使用类似

```text
2x1o + 1x2o + 1x3o
```

的 node output，再转换成 Cartesian/Kelvin (3\times6) block。

施加平移 gauge：

[
U_\eta\leftarrow
(I-TT^T)U_\eta
]

或质量加权版本。

监督方式：

* **179 strict**：
  [
  U_\eta^\star
  ============

  \mathcal D(\Phi^\star)\Lambda^\star
  ]
  全量监督；
* **610 branch samples**：
  [
  L_{\rm macro-U}
  ===============

  \rho!\left(
  \left|
  \frac{c_e}{\Omega}
  Z^{\star T}U_{\eta,\theta}
  --------------------------

  e^{ion,\star}
  \right|_{\rm Kelvin}
  \right);
  ]
* **factor consistency**：
  用独立 (\Phi_\theta,\Lambda_\theta,U_{\eta,\theta}) 施加 equilibrium/normal-equation residual，而不是 pseudoinverse chart；
* **inference**：
  主 ionic prediction 使用 (Z_\theta^{*T}U_{\eta,\theta})，完整 (\Phi,\Lambda) 保留为 auxiliary/audit heads。

这能立刻回答：问题究竟是学习 internal displacement response，还是学习完整 Hessian 后求逆。

## 4.2 更物理的低秩 response-subspace 模型

在质量加权坐标中，令

[
S=[\widetilde Z,\widetilde\Lambda]
\in\mathbb R^{d\times9},
\qquad
X=K^{-1}S.
]

由于 (S) 只有 9 列，

[
\operatorname{rank}(X)\le 9.
]

因此对 piezo-only，(K^{-1}\widetilde\Lambda) 的响应子空间维数至多为 6；对 piezo+dielectric 联合问题至多为 9。这是一个**对有限 RHS 数目严格低秩**的事实，而不是“假设最低几个 phonon modes 足够”。

令网络输出 (r) 个原子向量场：

[
W_\theta\in\mathbb R^{d\times r},
]

先投影平移，再等变正交化：

[
V_\theta=
\bar W_\theta
(\bar W_\theta^T\bar W_\theta+\epsilon I)^{-1/2}.
]

输出或构造：

[
H_\theta\in\mathbb R^{r\times r},
\quad
B_\theta=V_\theta^T\widetilde Z_\theta\in\mathbb R^{r\times3},
\quad
G_\theta\in\mathbb R^{r\times6}.
]

其中：

* (V)：node-level `r x 1o`；
* (H)：graph-level (r(r+1)/2) 个 `0e`，static 路径取 SPD；
* (B)：可由 (Z^*,V) 导出，也可作为 `r x 1o` auxiliary；
* (G)：每一 latent row 是 symmetric strain tensor，即 `0e + 2e`。

宏观响应为

[
\boxed{
e^{ion}
=

\frac{c_e}{\Omega}
B_\theta^T
H_\theta^{-1}
G_\theta
}
]

并可共享得到

[
\chi^{ion}\propto B^TH^{-1}B,
\qquad
\Delta C^{ion}\propto G^TH^{-1}G.
]

该模型在 latent 变换

[
V\to VR,\quad
H\to R^THR,\quad
B\to R^TB,\quad
G\to R^TG
]

下完全不变，因此不需要 mode matching。

## 4.3 若要 mode-resolved 解释，应监督 cluster contribution

对质量加权 eigenmode

[
K\xi_m=\kappa_m\xi_m,
]

定义

[
z_m=\widetilde Z^T\xi_m\in\mathbb R^3,
\qquad
g_m=\xi_m^T\widetilde\Lambda\in\mathbb R^6,
]

则

[
e^{ion}
=

\frac{c_e}{\Omega}
\sum_m f(\kappa_m)z_mg_m^T.
]

单个模式的符号会同时改变 (z_m,g_m)，乘积不变。对简并 cluster (C)，应使用

[
E_C=
\frac{c_e}{\Omega}
(\widetilde Z^TV_C),
f(V_C^TKV_C),
(V_C^T\widetilde\Lambda),
]

它在 (V_C\to V_CR) 下不变。

因此最合理的 mode supervision 是：

* cluster projector；
* cluster effective-charge Gram；
* cluster strain-coupling Gram；
* cluster (3\times6) response contribution。

而不是逐 eigenvector MSE。

---

# 5. 149 full factors + 4,961 total labels 的可靠训练范式

## 5.1 首先修正 exposure，而不是继续讨论 loss 权重

稿件所述 full-corpus replay 只有 100 structural、50 factor、100 joint **optimizer updates**。

定义任务 (t) 的有效数据遍历数

[
E_t=\frac{U_tB_t}{N_t}.
]

若仍使用历史 batch size 4，则：

[
E_{\rm macro}
=

\frac{100\times4}{4961}
\approx0.081,
]

而 50 factor updates 对 149 strict 样本也只有

[
\frac{50\times4}{149}\approx1.34
]

次遍历。历史配置确实记录过 batch size 4，但 full-corpus replay 的实际 batch 必须单独报告。

更严重的是，若 joint batch 从 4,961 中均匀采样，100 个 batch 中 strict 样本的期望出现次数为

[
100B\frac{149}{4961}.
]

当 (B=4) 时只有约 12 次 sample occurrences。

所以目前只能写：

> 增加 total 数据在当前 fixed-update exposure 下没有改善结果。

不能写：

> 增加 total 数据没有帮助。

可靠协议应为三条独立 dataloader：

* macro stream：4,961；
* branch/DFPT stream：610；
* strict stream：149；

每个“联合 epoch”分别保证三条流达到预定有效遍历数，而不是从并集随机抽样。

## 5.2 推荐的分阶段双塔范式

### Stage 0：结构与曲率表示预训练

优先使用 energy/force/stress + direct Hessian/HVP supervision，而不是只做 100 次 masked-structure update。PFT 的结果表明，仅通过 energy/force 间接约束 curvature 不够，直接采样 Hessian columns/HVP 能显著改善 force constants，而且计算可从全 Hessian 的二次开销降到单列 HVP。([arXiv][4])

### Stage 1：physical branch

仅用 610/149 数据训练：

[
Z^*,\quad
\Phi_{\rm raw},\quad
K_{\rm stat},\quad
U_\eta\ \text{或 reduced }(V,H,B,G).
]

使用真实 (Z^*,\Phi) 的 oracle substitution 保持每个子任务可解释。

### Stage 2：branch decomposition

在 610 上训练：

[
e^{ion},\quad e^{el},\quad
e^{el}+e^{ion}=e^{tot}.
]

此阶段 physical decoder 已有稳定初始化。

### Stage 3：large-total macro training

使用 4,961 total labels，但：

* total-only 梯度不得进入 factor decoder；
* 只更新 macro adapter/direct total head；
* 或只共享最底层的一两层几何表示，高层分别使用 physical adapter 与 macro adapter。

这避免 total-only 数据利用 branch gauge 把 ionic branch 压到零。

### Stage 4：小规模 joint calibration

只在 610 branch-labeled 样本上解冻少量共享层。是否解冻由 train 内 nested CV 决定，而不是 frozen test20。

## 5.3 partial-label likelihood

每个样本只对已观测量贡献 likelihood：

[
\begin{aligned}
\mathcal L_n
=&-\log p(e_n^{tot}\mid x_n)\
&-m_n^{branch}\log p(e_n^{ion},e_n^{el}\mid x_n)\
&-m_n^Z\log p(Z_n^*\mid x_n)
-m_n^\Phi\log p(\Phi_n\mid x_n)\
&-m_n^U\log p(U_{\eta,n}\mid x_n).
\end{aligned}
]

建议使用按 irrep block 定义的 heteroscedastic Student-(t) likelihood。这里的 mask 是观测模型的一部分，而不是把缺失 factor 伪装成零标签。

## 5.4 teacher/student 的安全用法

当前 ionic teacher 的 OOD skill 为负，不应给 4,351 个 total-only 样本生成 (Z,\Phi,\Lambda) pseudo-label。

可接受的 teacher/student 用法是：

* distill shared invariant/equivariant representations；
* 对旋转、cell basis change、small perturbation 做 consistency；
* 用 teacher ensemble uncertainty 决定新 DFPT acquisition；
* 只有当 physical teacher 在 nested formula-disjoint CV 上稳定正 skill 后，才考虑软 pseudo-label，并保留不确定度。

## 5.5 strict completion 的 missingness 不是随机的

strict label 是否可获得取决于：

* 空间群；
* printed-block coverage；
* 原子数；
* completion rank；
* closure gates。

因此它很可能是 missing-not-at-random。masked loss 默认训练的是“容易完成的高对称材料分布”，不一定是 4,961 条总体分布。

应训练一个 completion-propensity classifier

[
q_\psi(m_{\rm strict}=1\mid x)
]

并报告 AUC、晶系、原子数和 response magnitude 分层。最可靠的修正不是大幅 inverse-propensity weighting，而是主动补充 low-propensity、低对称和高响应材料。

---

# 6. 相关工作的直接启示

DFPT 的标准框架本来就是由 displacement、strain 和 electric field 的混合二阶导数组成，并强调不同电学/力学边界条件下响应定义不同；PiezoJet 的因子化物理基础是正确的，但不稳定参考结构和正则化算子的语义必须单独声明。([APS Link][5])

JARVIS 的高通量工作提供了 5,015 个非金属材料的 (\Gamma)-point phonon、BEC、piezo 和 dielectric 数据，是当前数据路径的主要基础。([Nature][6])

GMTNet 和 EATGNN 表明完整压电张量可由等变模型直接预测；这支持保留 direct observable baseline，但其随机拆分、重采样和误差定义不能直接与当前 formula-disjoint TRS 比较。([Proceedings of Machine Learning Research][1])

Falletta 等人的 electric-enthalpy 模型从统一标量势导出 polarization、BEC 和 polarizability，说明 derivative compatibility 是有价值的；但它同时直接监督相关电响应量，不能被解读为“仅靠能量结构先验即可识别所有高阶响应”。([Nature][7])

Equivar 等工作说明 BEC 本身适合用等变张量网络学习；这与 PiezoJet 中 BEC 已经不是主要瓶颈的诊断一致。([Nature][8])

Fang 等人的 equivariant Hessian/phonon 模型和 PFT 表明 curvature 需要专门的 Hessian supervision；后者特别提供了 stochastic Hessian-column/HVP 的可扩展实现路径。([arXiv][9])

Takigawa 等人的 factorized ionic dielectric 模型是最相关对照：它用等变 BEC 模型配合成熟的 pretrained ML potential 产生 phonons，并只在预测稳定的材料上评估静态逆。它没有额外的 (\Lambda) 因子；其 BEC 数据约 928 个氧化物，并在 839 个预测稳定氧化物上评价 ionic dielectric。因此它的成功并不能推出 149 个完整 (\Lambda) 足以训练 piezo factorization，反而显示“成熟 phonon prior + 稳定子集 + 无 strain-coupling 因子”与当前问题的差异。([arXiv][2])

极性材料的 (\Gamma) 点非解析项依赖 (q\to0) 的方向；Phonopy 也明确要求通过 `Q_DIRECTION` 指定 NAC 方向。未来应将短程 analytic force constants 与由 (Z^*,\epsilon_\infty) 构造的长程项分离，但在确认 VASP/JARVIS 源数据采用的边界条件前，不应盲目把 NAC 加入当前 homogeneous piezo solve。([APS Link][10])

---

# 7. 按优先级排序的行动清单

## P0-1：删除同-(A) pseudoinverse chart 作为生产参数化

**数学动机：** 满行秩时 (A\hat\Lambda=y) 恒成立，consistency 恒零，并产生 ghost gradient。

**所需数据：** 无新增数据。

**代码位置：** 构造式（30）的 lift 模块、factorized ionic forward、consistency loss。

**最小可证伪实验：**

1. 固定随机满行秩 (A,y,R)；
2. 计算当前 forward；
3. 对 (A) 做微小参数扰动，每次重算 chart；
4. 比较 autograd JVP 与实际 finite difference。

当前实现应出现：

[
\text{finite-difference}\approx0,\qquad
\text{autograd JVP}\neq0.
]

替换为独立

[
\Lambda_\theta^{ind}
\quad\text{或}\quad
U_{\eta,\theta}^{ind}
]

后，要求 JVP 与 finite difference 相对误差 (<10^{-4})。

**成功判据：** factorized 与 direct ionic 初始预测不再恒等；consistency 有非零且可下降的残差；梯度通过有限差分。

**失败判据：** factorized forward 仍能由代数恒等式化成 direct head，或梯度仍依赖 pseudoinverse chart/cutoff。

---

## P0-2：做 exposure-matched replay

**数学动机：** fixed updates 不能代表固定数据遍历量，当前实验可能严重 undertrained。

**所需数据：** 现有 4,961/610/149。

**代码位置：** trainer、sampler、日志。

**最小实验：** 每个 stream 分别做

[
1,\ 5,\ 10,\ 20
]

个 effective passes；记录 unique sample coverage、每类标签出现次数、train/val learning curve。direct baseline 与 physical model 使用相同 macro exposure，但 physical stream 额外保证相同 factor passes。

**成功判据：**

* 若 5–10 passes 后 train/val 已平台且 ionic 仍塌缩，才支持结构/监督瓶颈；
* 若随 passes 明显单调改善，则原“扩大 total 数据无效”的结论被证伪。

**失败判据：** 仍以 optimizer updates 而不是 examples/passes 比较模型。

---

## P0-3：审计光学算子、(Q) 定义和 VJP

**数学动机：** 若 (Q_o) 只是光学基，正确公式应为

[
\Phi_o=Q_o^T\Phi Q_o
=V\operatorname{diag}(\lambda)V^T,
]

[
\mathcal D_\delta
=

Q_oV\operatorname{diag}(f_\delta(\lambda))
V^TQ_o^T.
]

不能写成仅含 (Q_o\operatorname{diag}(f(\lambda))Q_o^T)，除非 (Q_o) 已包括 eigenvectors。

**所需数据：** 合成矩阵及现有 DFPT。

**代码位置：** reduced optical solver。

**最小实验：**

* 随机 (R\in O(d))，替换 (Q_o\to Q_oR)；
* 比较 forward 和 VJP；
* 测试重复、近重复 eigenvalues；
* 比较 eig 实现和
  [
  (\Phi_o^2+\delta^2I)X=\Phi_oB
  ]
  的 float64 solve。

**成功判据：** basis-change forward 相对差 (<10^{-10})，VJP finite-difference 相对差 (<10^{-4})，无 near-degeneracy NaN。

**失败判据：** 输出依赖任意 optical basis 或 float32 下残差随条件数爆炸。

---

## P0-4：统一 Kelvin metric 和 branch target space

**数学动机：** engineering Voigt 下，未加权 (3\times6) Frobenius 不是 Cartesian 旋转不变量。应使用

[
M_V=\operatorname{diag}(1,1,1,2,2,2),
]

[
|T|_{\rm cart}^2
=

\operatorname{tr}(TM_VT^T),
]

或先转换到完整 Cartesian。逐 component SmoothL1 即便在 Cartesian 中也通常不是旋转不变量。

**所需数据：** 无新增。

**代码位置：** total、ionic、(\Lambda)、BEC、(\Phi) losses；Reynolds projection preprocessing。

**最小实验：** 随机旋转结构、预测和标签，验证所有 loss 数值不变；同时对 610 样本统一使用

[
P_Ge^{el}+P_Ge^{ion}=P_Ge^{tot}.
]

**成功判据：** float64 随机旋转 loss 相对差 (<10^{-8})；branch-sum residual 接近机器精度。

**失败判据：** loss 随坐标轴改变，或 raw branch 与 projected total 仍属于不同 target space。

---

## P0-5：完成 operator/factor substitution grid

**数学动机：** 精确分离 (Z,\Phi,\Lambda/U) 和 operator 的误差。

**所需数据：** 179/610。

**代码位置：** evaluator。

**最小实验：** 对每个样本计算：

[
(Z^\star,\Phi^\star,\Lambda_\theta),
\quad
(Z^\star,\Phi_\theta,\Lambda^\star),
\quad
(Z_\theta,\Phi^\star,\Lambda^\star),
\quad
(Z_\theta,\Phi_\theta,\Lambda_\theta).
]

同时记录：

[
r_\delta
=

\frac{|Z^T\mathcal D_\delta(\Phi)\Lambda|}
{|Z^T\Phi^{-1}\Lambda|}
]

在 true-stable 和 predicted spectrum 上的分布，以及 (\delta/10,\delta,10\delta) 敏感性。

**成功判据：** 能以 paired per-material 指标明确归因。若 true (Z,\Phi)+predicted (\Lambda) 仍贡献绝大部分误差，立即转向 (U_\eta)。

**operator 失败判据：** 超过一半 active 样本有 (|\lambda|<3\delta)，且改变 (\delta) 一 decade 使响应幅度变化超过 20%。

---

## P1-1：实现独立 direct-(U_\eta) head

**数学动机：** 直接学习决定 ionic piezo 的原子内部位移响应，绕开病态 inverse 和完整 (\Lambda) nullspace。

**所需数据：**

* 179 full (U_\eta^\star)；
* 610 true-(Z^*) macro constraints；
* 现有 partial (\Lambda) 继续作为 auxiliary。

**代码位置：** node equivariant output、acoustic projection、ionic forward。

**最小实验：** 与独立 raw-(\Lambda) head 在完全相同 encoder/exposure 下做 nested formula-disjoint 5-fold CV。

**成功判据：**

* 五折中至少四折 paired ionic skill 改善；
* hierarchical bootstrap CI 高于零；
* held-out amplitude ratio 从 0.036 提升到至少 0.20；
* material-macro cosine 至少达到 0.20，且不靠 micro aggregation。

**失败判据：** amplitude 仍 (<0.1)、cosine 不稳定，说明结构表示或标签覆盖仍不足。

---

## P1-2：质量加权 + scale–shape 预条件

**数学动机：** 去除原子质量和跨材料刚度尺度造成的条件数差异，显式学习 response amplitude。

**所需数据：** 现有因素标签和原子质量。

**代码位置：** factor preprocessing、head output transform、operator。

**最小实验：** 比较 raw Cartesian 与 mass-weighted/relative-(\epsilon) 两种参数化；保持网络和训练样本完全相同。

**成功判据：** 低模误差、seed variance 和 amplitude bias 同时下降；不同材料的 standardized factor RMS 不再跨多个数量级。

**失败判据：** 只改善 factor component MAE，不改善任何 response-active metric。

---

## P1-3：实现 (r=6/9) response-subspace 模型

**数学动机：** piezo RHS 只有 6 列，联合 piezo+dielectric 只有 9 列，响应解天然低秩。

**所需数据：** 179 完整因子即可先做 oracle。

**代码位置：** latent vector-field head、equivariant orthogonalization、reduced (H,B,G) solver。

**最小实验：**

1. 用 true (U_\eta) SVD 构造 oracle (V_r)；
2. 测试 (r=2,4,6) 的 response reconstruction；
3. 联合 dielectric 时测试 (r=6,9,12)。

**成功判据：**

* piezo oracle (r=6) 重建 (U_\eta) 相对误差接近数值精度；
* 带 reduced physical solve 的 (r=6/9) 宏观响应重建率 (>95%)；
* learned model 超过 direct-(U) 或以更少数据达到相同 skill。

**失败判据：** oracle 本身不能重建，说明实现、坐标语义或 target construction 有错误；不是增加 (r) 就能掩盖的问题。

---

## P1-4：physical/macro 双塔与 partial-label likelihood

**数学动机：** total-only 梯度不能识别 ionic factors，并会利用 branch gauge 破坏 physical decoder。

**所需数据：** 现有全部数据。

**代码位置：** encoder adapters、optimizer parameter groups、masked likelihood。

**最小实验：**

* physical decoder 只受 610/149 更新；
* macro adapter 受 4,961 更新；
* 在 610 上做最后 branch calibration；
* 与普通 fully shared joint training paired 比较。

**成功判据：** total-only batch 对 factor decoder 的 gradient norm 精确为零；physical CV skill 不再在 macro stage 后下降；total TRS 至少不低于 matched direct baseline。

**失败判据：** 分塔后 physical skill 仍同样塌缩，说明主要问题是信息覆盖而非负迁移。

---

## P1-5：获取直接的 strain-force 或 relaxed-displacement 标签

**数学动机：** 当前真正缺少的是 (\Lambda) 或 (U_\eta)，不是更多 total tensors。

**所需数据：** 先做 50–100 个 formula-diverse pilot，随后 200–500 个。

固定内部坐标下：

[
\Lambda_{:,\mu}
\approx
\frac{F(+h_\mu)-F(-h_\mu)}{2h}.
]

完全 relaxation 后：

[
U_{:,\mu}
\approx
\frac{u_{\rm rel}(+h_\mu)-u_{\rm rel}(-h_\mu)}
{2h}.
]

**代码位置：** VASP workflow、atom mapping、finite-difference validator、cache schema。

**最小实验：** 优先采集 low-symmetry、高 ionic response、soft-stable 及当前 completion propensity 低的材料；不要继续只优化 strict-completion 成功率。

**成功判据：** 随新增 full-(U/\Lambda) 数量的 nested-CV learning curve 有稳定正斜率，且不同晶系均改善。

**失败判据：** 新增 100–200 个多样标签后仍无 active-response improvement；此时才应重点怀疑 local representation 或 DFPT source semantics。

---

## P2：边界条件、有限温和概率主动学习

**数学动机：**

* 极性材料需要明确 analytic (\Gamma)、fixed-(E)/fixed-(D) 及 NAC 语义；
* unstable saddle 不存在普通零温静态响应；
* high-response tail 的不确定性主要来自相关的 mode/factor posterior。

**所需数据：**

* (Z^*,\epsilon_\infty)、小 (q) phonons；
* 部分 SSCHA/TDEP reference；
* physical-model ensemble。

**代码位置：** long-range force-constant module、dataset metadata、posterior decoder、acquisition queue。

**最小实验：**

* 对少量强极性材料比较 analytic (\Gamma) 与方向依赖 NAC；
* 对温度稳定相比较 raw Hessian 与 TDEP/SSCHA free-energy Hessian；
* 用 ensemble 选择最大预期 ionic-response information gain 的新材料。

**成功判据：** 改善集中在预先定义的 polar/unstable strata，且不损害普通 stable nonpolar 材料。

**失败判据：** 对所有材料统一加 NAC 或 finite-T correction 才能改善，通常意味着边界条件或数据管线仍有未识别混用。

---

# 最终排序

我建议严格按以下顺序执行：

[
\boxed{
\text{chart 恒等退化}
\rightarrow
\text{exposure replay}
\rightarrow
\text{算子/metric 审计}
\rightarrow
\text{direct }U_\eta
\rightarrow
\text{mass/scale preconditioning}
\rightarrow
\text{response subspace}
\rightarrow
\text{新增 strain-response 数据}
}
]

前三项不需要新增任何 DFPT 计算，就能判断当前 3.7% amplitude ratio 究竟是计算图退化、训练覆盖不足，还是物理中间量不可泛化。就现有证据看，我不会把主要责任归给 (\delta)，也不会接受“factorization 已被证伪”的表述；更准确的结论是：

> 当前 factorized forward 在满行秩区间并非独立预测，fixed-update full-corpus replay 也尚不足以检验数据规模效应；在修复这两点后，最有希望且最可证伪的下一模型是独立、平移无关、等变的 (U_\eta=\partial u/\partial\eta) head，以及由至多 6–9 个响应向量场构成的 reduced optical-response subspace。

[1]: https://proceedings.mlr.press/v235/yan24d.html "https://proceedings.mlr.press/v235/yan24d.html"
[2]: https://arxiv.org/pdf/2509.26022 "https://arxiv.org/pdf/2509.26022"
[3]: https://arxiv.org/abs/2103.03973 "https://arxiv.org/abs/2103.03973"
[4]: https://arxiv.org/html/2601.07742v1 "https://arxiv.org/html/2601.07742v1"
[5]: https://link.aps.org/doi/10.1103/PhysRevB.72.035105 "https://link.aps.org/doi/10.1103/PhysRevB.72.035105"
[6]: https://www.nature.com/articles/s41524-020-0337-2 "https://www.nature.com/articles/s41524-020-0337-2"
[7]: https://www.nature.com/articles/s41467-025-59304-1 "https://www.nature.com/articles/s41467-025-59304-1"
[8]: https://www.nature.com/articles/s41598-025-01250-5 "https://www.nature.com/articles/s41598-025-01250-5"
[9]: https://arxiv.org/html/2403.11347v1 "https://arxiv.org/html/2403.11347v1"
[10]: https://link.aps.org/doi/10.1103/PhysRevB.55.10355 "https://link.aps.org/doi/10.1103/PhysRevB.55.10355"
