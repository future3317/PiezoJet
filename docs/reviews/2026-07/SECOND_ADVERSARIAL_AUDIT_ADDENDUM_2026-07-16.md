# PiezoJet 对抗式二次审查增补

日期：2026-07-16
范围：在既有审计与 direct-\(U_\eta\) 修正基础上，复核 response-active
subspace、signed resolvent、detached Moore--Penrose chart、低秩架构、半监督、
有效曝光量、质量加权及 latent covariance 等竞争性解释。本增补不重写原审计。

## 0. 结论先行

当前最可靠的裁决是：

1. **detached Moore--Penrose chart 的问题是代数定理，不是待调超参数。**在
   \(A\) 满行秩且 forward 中的 \(A\) 与 detached \(\bar A\) 数值相同时，
   \(A\hat\Lambda=y\) 恒成立；该路径把 factorized ionic forward 退化成 direct
   ionic head，并制造 chart-dependent straight-through/ghost gradient。生产代码已
   删除该 chart、ridge lift、active/null projector 和相关 fallback。
2. **direct-\(U_\eta\) 是当前可辨识性最好的物理候选，但尚无性能证据。**它用
   149 个 strict labels 直接监督
   \(U_\eta^\star=\mathcal D_\delta(\Phi^\star)\Lambda^\star\)，并以
   \(Z^{*T}U_\eta\) 产生 ionic response；one-pass seed-42 smoke 的 ionic skill
   仍接近零，不能宣传为改进。
3. **“resolvent 通过预测极软模压低响应”不解释当前 smoke。**schema-6 test20
   中 true 与 predicted 共各 480 个 optical modes，全部位于
   \(|\lambda|\ge3\delta\)。这只反驳当前 checkpoint 的 soft-mode shortcut，
   不消除训练轨迹中的大斜率风险，也不排除把谱推向硬区的幅度压缩。
4. **response-active alignment 是得到新增支持的诊断解释，但还不是因果结论。**
   true-BEC 下 predicted direct-\(U_\eta\) 的 \(Z^{*T}U\) 平均方向 cosine 为
   0.1153、幅度比为 0.0563；19 个非零预测材料的位移列空间 projector overlap
   为 0.6471，最小 principal cosine 为 0.5192；另有 1 个材料预测
   \(U_\eta\) 数值秩为 0。方向、幅度和子空间均有问题，但 one-pass 的 BEC、
   \(\Phi\) 与 \(U_\eta\) 也都未学好，不能把失败唯一归因于 mode alignment。
5. **\(\operatorname{rank}(U_\eta)\le6\) 只是六个 strain RHS 导致的矩阵秩
   上界，不是“六个 phonon eigenmodes 主导”的证据。**test20 的 true target
   rank 分布为 rank3/4/5/6 = 4/2/2/12；rank-4 SVD 保留 95.98% 位移奇异能量，
   但 ionic response 相对误差仍为 19.81%。rank-6 精确重建是线性代数恒等事实。
6. **下一轮的主实验应是 exposure-matched direct-\(U_\eta\) 对 matched direct
   total，而不是新增 fallback、mode slots 或 factor pseudo-labels。**代码已能记录
   每条数据流的 unique coverage、examples seen、effective passes、optimizer
   updates 和各类标签进入梯度的 update 数；尚需真实执行 1/5/10/20 passes ×
   3 seeds。

证据标签：**[代数]** 可直接证明；**[代码]** 已由源码/测试确认；**[实验]** 已有
固定产物支持；**[假设]** 尚需最小实验；**[方向]** 仅为后续研究选项。

## 1. 与此前审计一致的结论

### 1.1 宏观响应不能识别原子级 \(\Lambda\) [代数]

对固定 \(Z^*,\Phi\)，定义

\[
A=\frac{c_e}{\Omega}Z^{*T}\mathcal D_\delta(\Phi),\qquad
e^{\rm ion}=A\Lambda .
\]

\(A\in\mathbb R^{3\times3N}\)，而 \(\Lambda\in\mathbb R^{3N\times6}\)。
即使 \(A\) 满行秩，\(\ker A\) 通常仍为高维。仅凭 18 个宏观 ionic 分量不能
恢复完整原子级 \(\Lambda\)。full-\(\Lambda\) strict labels 和 printed-block
labels 是不可替代的锚点。

### 1.2 total-only 数据还存在 electronic/ionic branch allocation gauge [代数]

若只监督

\[
e^{\rm total}=e^{\rm elec}+e^{\rm ion},
\]

则任意张量 \(Q\) 下
\((e^{\rm elec}+Q,e^{\rm ion}-Q)\) 给出同一 total。当前双塔和同 OUTCAR
branch labels 是正确处理：4,961 条 total-only 样本只更新 macro tower；物理
factor heads 不接受 total-only 梯度。

### 1.3 不应逐个匹配简并/近简并 eigenvectors [代数/代码]

单 eigenvector 有 sign gauge，\(r\) 重简并子空间还有 \(O(r)\) gauge。可审计量
应是 projector overlap、principal angles、cluster contribution 或
cross-covariance。已有 low-mode evaluator 使用
\(\|V_q^T\widehat V_q\|_F^2/q\)，本轮又为 direct-\(U_\eta\) 增加列空间
projector 与 \(Z^{*T}U\) 诊断；没有新增逐 mode pairing。

### 1.4 不应通过固定谱归一化“消除 gauge” [代数]

\(Z^*\)、\(\Phi\)、\(\Lambda\) 都带真实物理单位和材料幅度。强行固定
\(\|Z^*\|\)、\(\operatorname{tr}\Phi\) 或 \(\|\Lambda\|\) 会把真实材料差异
误当 gauge 抹除。scale--shape 分解只能作为保持单位、可逆重构的参数化实验，
不能改变监督目标。

### 1.5 detached chart 的完整推导与无训练 JVP 判据 [代数]

历史定义为

\[
\bar A=\operatorname{stopgrad}(A),\qquad
\widehat\Lambda=\bar A^+y+(I-\bar A^+\bar A)R.
\]

当 \(A\) 满行秩且与 \(\bar A\) 数值相同，Moore--Penrose 恒等式给出

\[
A\bar A^+=I,
\qquad
A(I-\bar A^+\bar A)=A-A\bar A^+\bar A=A-\bar A=0,
\]

因此

\[
A\widehat\Lambda=y
\]

恒成立。factorized--direct consistency 在该区域恒为零，所谓 factorized forward
并不是独立预测。更严重的是，autograd 把 \(\widehat\Lambda\) 视为不依赖 \(A\)，
所以给出 \(d(A\widehat\Lambda)=dA\,\widehat\Lambda\)；真实扰动若每次重算
pseudoinverse/chart，则 \(A\widehat\Lambda(A)=y\) 的总导数为零。

无需训练的最小复现应使用 float64 随机满行秩
\(A\in\mathbb R^{3\times n}\)、\(y\in\mathbb R^{3\times6}\)、
\(R\in\mathbb R^{n\times6}\) 和随机方向 \(H\)：

1. autograd JVP 使用 detached \(\bar A\) 构造一次 chart；
2. 中心 finite difference 在 \(A\pm hH\) 处分别重算 pseudoinverse/chart；
3. 检查 forward closure \(\|A\widehat\Lambda-y\|/\|y\|<10^{-10}\)；
4. 检查重算-chart finite-difference norm \(<10^{-8}\)，而 detached JVP norm
   \(>10^{-6}\)，两者相对差异 \(>0.9\)。

该测试用于说明已删除历史路径的数学问题；生产代码不应为运行它而重新引入
chart。相应地，附件要求监控 predicted \(A_\theta\) 的奇异值、rank 和 SVD cutoff
只对该历史 chart 有意义，对当前 direct-\(U\) 生产模型不再是必要运行指标。

## 2. 对此前审计的重要补充

### 2.1 signed regularized resolvent 的完整裁决

当前生产算子为

\[
f_\delta(\lambda)=\frac{\lambda}{\lambda^2+\delta^2},\qquad
f_\delta'(\lambda)=\frac{\delta^2-\lambda^2}
{(\lambda^2+\delta^2)^2}.
\]

- 它连续、光滑、有界、奇且符号保持，不存在跨零点的函数 jump。
- \(f_\delta'(0)=1/\delta^2\)，因此零点附近可能产生谱梯度刚性。这是
  optimization/conditioning 风险，不是 forward 不连续。
- \(f_\delta(\lambda)\to0\) 同时发生在 \(\lambda\to0\) 与
  \(|\lambda|\to\infty\)，所以理论上存在 soft-end 和 hard-end 两种幅度 shortcut。
- 任意连续、奇、符号保持且有界的滤波器都必须满足 \(f(0)=0\)。不存在既连续
  穿过零点、又保留 \(1/\lambda\) 软模发散的简单 signed filter。
- 不采用 \(\operatorname{sgn}(\lambda)/(|\lambda|+\delta)\)：它在零点不连续、
  不可微。

新增 schema-6 证据：test20 的 true/predicted optical spectra 均为
0/0/480（\(<\delta\) / \([\delta,3\delta)\) / \(\ge3\delta\)）。因此当前
smoke 没有 soft-end shortcut 证据。由于只检查了一个 checkpoint，尚不能认定
resolvent 完全无关；训练轨迹三段占比、真实/预测 stiffness error 与 response
alignment 仍应联合记录。

### 2.2 response-active alignment 的新 projector/cross-covariance 证据

对 strict 样本，以 true BEC 的坐标列空间 projector \(P_Z\) 和 direct
displacement response 的列空间 projector \(P_U\) 计算 gauge-invariant 指标，
并显式比较

\[
C^\star=Z^{*T}U_\eta^\star,\qquad
\widehat C=Z^{*T}\widehat U_\eta .
\]

schema-6 one-pass test20 得到：

| 指标 | 结果 |
| --- | ---: |
| nonzero predicted-\(U\) 材料 | 19/20 |
| 列空间 projector overlap（19 个） | 0.6471 |
| 最小 principal cosine（19 个） | 0.5192 |
| true \(U\) 在 true-charge active space 的能量比例 | 0.2657 |
| predicted \(U\) 在 true-charge active space 的能量比例 | 0.2793 |
| \(P_Z\widehat U\) 与 \(P_ZU^\star\) 方向 cosine | 0.1191 |
| \(Z^{*T}\widehat U\) 与 \(Z^{*T}U^\star\) 方向 cosine | 0.1153 |
| \(Z^{*T}\widehat U\) 幅度比 | 0.0563 |

active energy fraction 接近不表示预测正确；它只表示两者把相近比例的总位移
能量放入 BEC-active 坐标空间。低 cross-covariance cosine 和 5.63% 幅度比才
直接显示 response-active 方向与尺度失败。最终使用 predicted BEC 时 ionic
幅度进一步降至 0.54%，说明 BEC 误差还会与 \(U\) 误差复合。

### 2.3 低秩 oracle 补充了“可压缩性”而非“mode 数”证据

对每个 true \(U_\eta^\star\in\mathbb R^{3N\times6}\) 做 SVD，test20 的均值为：

| rank | 位移相对 Frobenius 误差 | true-BEC response 相对误差 | 奇异能量保留率 |
| ---: | ---: | ---: | ---: |
| 1 | 0.7510 | 0.7520 | 0.4328 |
| 2 | 0.4729 | 0.5650 | 0.7453 |
| 4 | 0.1436 | 0.1981 | 0.9598 |
| 6 | \(1.3\times10^{-15}\) | \(1.9\times10^{-13}\) | 1.0000 |

rank-4 对位移已相当紧凑，但 response error 仍接近 20%，说明 response-relevant
方向不能仅按位移能量排序。dense reduced candidate 若继续，应优先优化/监督
\(Z^{*T}VC\) 或 generalized response-weighted oracle，而不是声称“四个/六个
phonon modes 已足够”。

### 2.4 有效曝光量现在可审计，但尚无新性能结果 [代码]

`train.py` 的 multistream 路径完整 exhaust macro/branch/strict 三条 DataLoader，
并输出：stream material count、unique samples seen、examples seen、factor/joint
optimizer updates、effective passes，以及 macro total、BEC、\(\Phi\)、printed
\(\Lambda\)、ionic/electronic/branch sum、full \(\Lambda\)、direct \(U\)、normal
equation 和 factor consistency 各自进入梯度的 update 数。multistream 模式禁止
update cap。尚未运行的新 replay 不能由代码功能替代。

## 3. 此前可能错误或证据不足之处

### 3.1 “\(\Phi\) 已不是瓶颈”证据不足

历史 true \(Z^*,\Phi\)+predicted \(\Lambda\) 的约 5% 幅度和近零 cosine 支持
当时的 predicted-\(\Lambda\) 方向问题；但当前 direct-\(U\) one-pass 中：

- BEC material-macro cosine = -0.0088；
- force-constant material-macro cosine = -0.7355；
- \(U_\eta\) material-macro cosine = 0.0346；
- full \(\Lambda\) material-macro cosine = -0.0756。

因此当前证据只能说“所有 learned physical factors 都未收敛”。不能把当前失败
继续单因归给 \(\Lambda\)，也不能宣布 \(\Phi\) 已排除。

### 3.2 “resolvent 是振幅塌缩主因”没有成立

现有谱三段统计反驳了当前 checkpoint 的 predicted-soft shortcut。硬区
\(f_\delta\sim1/\lambda\) 仍可能压幅，但必须证明预测 stiffness 系统性偏硬且
这种偏差与响应幅度相关。现在更准确的表述是：resolvent 是潜在放大器/条件数
风险；当前 one-pass 的直接证据主要是 factors 与 cross-covariance 未学好。

### 3.3 “rank \(\le6\) 推出六个物理 modes”是错误推论

rank 上界来自矩阵只有六列，不包含 \(\Phi\) eigenbasis 的稀疏性信息。mode-slot
模型额外假设共享的少量物理本征模主导，必须单独用 cluster response oracle
检验。

### 3.4 质量加权、scale--shape 与 joint latent covariance 仍是研究假设

质量加权
\(K=M^{-1/2}\Phi M^{-1/2}\)、
\(\widetilde Z=M^{-1/2}Z^*\)、
\(\widetilde\Lambda=M^{-1/2}\Lambda\)
是可逆坐标变换，可能改善 conditioning，但不会自动解决 OOD 子空间方向。当前
JARVIS cache 已审计 mass-unweighting；生产 target 仍是物理 Angstrom 坐标。只有
matched ablation 才能说明质量加权是否提高学习效率。

同样，
\(\mathbb E[Z^T\mathcal D(\Phi)\Lambda\mid x]\) 一般不等于条件均值乘积，
但“必须用 joint latent variable”尚无直接证据。先测量跨因子残差 covariance 和
calibration，再决定是否引入复杂 decoder。

## 4. 必须修正的数学/实现错误及状态

| 问题 | 裁决 | 状态 |
| --- | --- | --- |
| detached pInv chart 满行秩退化 | 数学错误；forward 恒等于 direct \(y\)，autograd 与重算 chart 的 finite difference 不一致 | 已从生产模型删除 |
| ridge lift 被称为 minimum-norm/null projector | \(\gamma>0\) 时只是 ridge lift，且残差算子不幂等 | 整条 lift 路径已删除，不再改名保留 |
| predicted \(\lambda_{\min}\) 硬切换 exact/regularized | 阈值不连续且允许错误分支 | 生产只接受 regularized；exact 仅 true-stable diagnostic |
| schema 缺少 predicted spectrum 三段统计 | 无法裁决 soft/hard shortcut | 已加入 schema 6 并重跑 test20 |
| rank-6 被误读为六个 phonon modes | 线性代数类别错误 | evaluator 文本与测试明确区分，并加入 rank oracle |
| response-active 假设只看单 eigenvectors | 受 sign/degeneracy gauge 污染 | 已加入 projector/principal-angle/cross-covariance 诊断 |
| exposure 只报 optimizer steps | 不同数据规模不可比 | 已加入完整 stream exposure 和 label-gradient-update 账本 |
| 配置注释仍把当前 completion 说成 v4/99 | 与实际 v7/179 不一致 | 已修正为 v7/179 与冻结 test20 |

## 5. 可验证但尚未成立的架构假设

### 5.1 direct-\(U_\eta\) 会提高 strict-label 样本效率 [假设]

可辨识性更好不保证泛化更好。它需要 3-seed exposure-matched replay 才能成立。
若只改善训练集 full-\(U\) 而 holdout ionic skill 仍不为正，则说明 head/encoder
仍未学到跨化学式的 response-active covariance。

### 5.2 low-rank \(U=VC\) 会提高泛化 [假设]

该参数化自动满足 rank \(\le r\) 且能使用完整 atom-level \(U^\star\) 标签。
但 \(V\to VR, C\to R^{-1}C\) 有 latent basis gauge，需要通过 QR/polar gauge 或
只监督重构量处理。rank-4 oracle 的 19.8% response error 已表明 \(r=4\) 不是
无损选择；\(r=6\) 应作为第一个不牺牲表达能力的候选。

### 5.3 dense reduced \((V,H,G)\) 会优于 direct \(U\) [假设]

\[
B=V^T\widetilde Z,\quad G=V^T\widetilde\Lambda,\quad
e^{\rm ion}=\frac{c_e}{\Omega}B^T H(H^2+\delta^2I)^{-1}G.
\]

它在 latent basis rotation 下可保持不变，并比 diagonal slots 更能表示近简并
mode mixing；但同时重新引入 reduced inverse 的谱 conditioning。必须先用 true
factors 做 held-out oracle，再写生产 head。

### 5.4 mass-weighted scale--shape head 会改善 conditioning [假设]

可显式写
\(\widetilde Z=s_Z\bar Z, K=s_K\bar K,
\widetilde\Lambda=s_\Lambda\bar\Lambda\)，保留真实宏观尺度
\(s_Zs_\Lambda/s_K\)。尚需证明 shape 与 log-scale 的残差统计更平稳，并确保
反变换后物理单位、声学投影和 response closure 不变。

### 5.5 joint latent covariance 能修复均值乘积偏差 [方向]

这是合理的概率建模方向，但 149 strict labels 对高维 joint decoder 可能不足。
只有在 deterministic direct-\(U\) exposure curve 显示稳定但受系统 covariance
偏差限制后，才值得进入实现。

## 6. 暂不建议实施的高风险方案

1. **factor pseudo-labeling 其余 4,961 条样本。**当前 teacher 没有正的 OOD
   ionic skill，会复制 amplitude collapse、high-symmetry selection bias、错误
   orientation 和 branch allocation bias。可以蒸馏底层 representation 或做
   equivariance consistency，不能生成 \(Z^*,\Phi,\Lambda\) 伪真值。
2. **diagonal physical mode slots。**它额外引入 slot permutation、sign 和
   degeneracy \(O(r)\) gauge；Softplus stiffness 又与保留 source negative modes
   冲突。dense reduced oracle 之前不实施。
3. **把 raw signed Hessian 与 SPD static Hessian 双分支直接塞进生产模型。**这
   需要明确有限温自由能/参考态语义和新标签；当前不能作为不稳定结构的 fallback。
4. **固定 factor norms/trace 的“gauge fixing”。**会破坏物理幅度与单位。
5. **用新的不连续 signed filter 或 predicted-spectrum gate。**会重新引入已删除
   的不连续与错误分支。
6. **只调 loss、PCGrad 或扩大模型后宣称解决。**这些不能推翻可辨识性、数据曝光
   或 response-subspace 假设，只有在 matched 最小实验之后才有解释价值。

## 7. 更新后的优先级行动清单

### P0-A：完成 exposure-matched direct-\(U\) vs direct-total replay

- **数学动机：**将“样本数”与“每类标签实际梯度曝光”分开，避免 fixed-update
  混淆。
- **所需数据：**formula-disjoint macro 4,961、branch 610、strict train149；冻结
  val10/test20。
- **代码位置：**`scripts/run_exposure_matched_replay.ps1`、
  `src/piezojet/train.py`、`src/piezojet/train_direct_baseline.py`。
- **最小实验：**1/5/10/20 effective passes，已注册 seeds 42/7/1729；每个条件保存
  exposure ledger、validation-selected checkpoint 和 schema-6 test report。
- **成功判据：**预注册主指标 direct-\(U\) ionic material-macro MAE skill 与 cosine
  在三 seed 均值上为正，且相对 matched direct-total 的 total TRS 不退化；报告
  seed spread，不按 test 选点。
- **失败判据：**ionic skill 仍不为正、仅单 seed 改善、或改善来自不等曝光/不同
  checkpoint 规则。

### P0-B：沿训练轨迹记录 response-active alignment 与谱三段占比

- **数学动机：**区分“列空间没对齐”“true-BEC active cross-covariance 错误”与
  “resolvent 谱 shortcut”。
- **所需数据：**strict train/val；test 只在最终 validation-selected checkpoint
  使用。
- **代码位置：**`src/piezojet/evaluate_dfpt.py` 已有 helper；训练 driver 在固定
  pass checkpoint 调用 evaluator。
- **最小实验：**对 P0-A 每个 pass/seed 的 val checkpoint 报 projector overlap、
  principal cosine、\(Z^{*T}\widehat U\) cosine/amplitude、三个谱区间。
- **成功判据：**alignment 指标随 validation ionic skill 稳定上升，并跨 seed
  重现；没有通过把谱推向 soft/hard 极端获得虚假幅度。
- **失败判据：**alignment 上升但响应不改善，或只有 test 上相关；此时不能把它
  当架构选择依据。

### P1-A：response-weighted low-rank \(U=VC\) 压缩 oracle

- **数学动机：**利用严格的 rank \(\le6\) 结构，同时避免把 RHS rank 误读为
  eigenmode sparsity。
- **所需数据：**179 strict completions，保持 frozen 149/10/20；true BEC 与
  regularized true \(U\)。
- **代码位置：**只扩展 `evaluate_dfpt.py` oracle；当前不在 `model.py` 增加
  rank-6 head。`r=6` 需要约 `18N+36` 个输出，相比 direct-`U` 的 `18N` 不压缩，
  反而增加 latent basis gauge。
- **最小实验：**比较 ordinary SVD 与以 \(\|Z^{*T}(U_r-U)\|_F\) 为目标的
  response-weighted rank 2/4/5 oracle；只有 `r<6` 在 validation folds 达到预注册
  response 重建阈值才考虑 learned compression。
- **成功判据：**某个 `r<6` 跨 fold 保持低 response error，且预计输出量确实少于
  `18N`。
- **失败判据：**只有 rank-6 trivial exact，或 rank-4/5 丢失关键 response-active
  信息；继续保留 full direct-`U`。

### P1-B：质量加权与 scale--shape 的只改坐标 ablation

- **数学动机：**改善不同元素质量/因子幅度造成的条件数，不删除物理尺度。
- **所需数据：**DFPT cache 中审计过的 masses、\(Z^*,\Phi,\Lambda,U\)。
- **代码位置：**`data.py` 提供可逆 target transform；`model.py` head 输出
  mass-weighted coordinates；`train.py` 在物理单位反变换后计算 closure。
- **最小实验：**raw Cartesian、mass-weighted、mass-weighted+log-scale/shape 三个
  matched seed-42 smoke；通过后才做三 seed。
- **成功判据：**训练/validation factor gradient dynamic range 和 factor MAE 改善，
  反变换 response closure 达到现有数值阈值，单位/声学投影测试全过。
- **失败判据：**只改善归一化 loss 而物理单位 ionic skill 不变，或破坏 source
  negative modes/closure。

### P2：测量残差 covariance，再决定 joint latent decoder

- **数学动机：**检验条件均值乘积近似是否产生系统偏差。
- **所需数据：**P0-A 三 seed 的 held-out factor residuals；不得用 test 调模型。
- **代码位置：**新增只读 analysis script，先不改 `model.py`。
- **最小实验：**在 val10 和交叉验证 train folds 上估计
  \(\Delta Z,\Delta\Phi,\Delta U\) 的 response-projected covariance 与 bootstrap
  uncertainty。
- **成功判据：**出现跨 fold、跨 seed 同号且量级显著的 covariance，并能解释
  deterministic response bias。
- **失败判据：**covariance 不稳定或置信区间覆盖零；不实现 latent decoder。

### P3：受限 teacher--student / active acquisition

- **数学动机：**利用 total-only 几何覆盖而不伪造不可辨识 factors。
- **所需数据：**4,961 structures、teacher ensemble uncertainty、新取得的真实
  DFPT labels。
- **代码位置：**只允许 encoder representation distillation、equivariance/
  perturbation consistency 与 acquisition ranking；physical decoder 仍只由
  branch/strict labels 更新。
- **最小实验：**representation-only student 对不使用 student 的 P0-A control；
  acquisition 先做 retrospective coverage audit。
- **成功判据：**三 seed OOD ionic skill 提升且 uncertainty 对新 label error 校准。
- **失败判据：**需要 factor pseudo-label 才改善，或 teacher 本身 OOD skill 非正。

## 8. 架构裁决表

| 方案 | 数学一致性 | 可辨识性 | 对 149 strict 的样本效率 | 负模式 | 完整 \(\Lambda\) 标签 | mode gauge | 推理复杂度 | 最关键可证伪实验 | 裁决 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| detached chart | forward 代数退化且 ghost gradient | 宏观 \(y\) 不能识别 \(\Lambda\) | 表面高，实际绕过 factor | 取决于外部算子 | chart residual 可吸收但无物理识别 | 无显式 mode gauge，存在 chart gauge | pInv/SVD | JVP vs 每次重算 chart 的 finite difference | **删除；已完成** |
| independent raw-\(\Lambda\) | 数学正确 | full label 下可识别；宏观下有大 response-null space | 低到中，高维 target | 可保留 | 可以直接监督 | eigenbasis 不显式出现 | \(O((3N)^3)\) 算子传播或 iterative apply | true \(Z,\Phi\)+pred \(\Lambda\) OOD response | **保留 factor diagnostic，不作唯一 ionic path** |
| direct-\(U_\eta\) | 与 regularized response 一致，无 inverse 梯度 | strict \(U\) 直接可识别；macro-only 仍有 \(Z/U\) 乘积 gauge | 当前最有希望 | target 继承 signed resolvent | 通过 \(U^\star\) 与 normal equation 利用 | 无 phonon-mode gauge | \(O(N)\) contraction | exposure-matched 3-seed replay | **当前生产候选** |
| dense reduced \((V,H,G)\) | 若保持 latent rotation invariance则正确 | 比 slots 好，但 \(V,H,G\) 内部仍有 basis gauge | 未知，可能更好 | signed \(H\) 可保留 | 可通过 reduced/full reconstruction 使用 | \(O(r)\) basis gauge | \(O(Nr+r^3)\) | true-factor response-weighted oracle 后 matched ablation | **先 oracle，暂不生产实现** |
| diagonal mode slots | 额外假设，近简并 mixing 表达受限 | slot permutation/sign/degenerate gauge 严重 | 未知且可能差 | Softplus 版本不能保留 | 无完整 \(V\) 时难监督 full \(\Lambda\) | 高 | \(O(NM+M)\) | cluster-response oracle 证明少数物理 modes 主导 | **暂不建议** |
| direct total baseline | 宏观张量预测一致 | total 可识别，branch/factors 不可识别 | 对 4,961 total labels 最高 | 不适用 | 不能利用原子级 \(\Lambda\) | 无 | 低 | 与 direct-\(U\) 完全 matched 的 total/ionic 指标 | **必须保留为控制组** |

## 9. 实现与产物索引

- evaluator：`src/piezojet/evaluate_dfpt.py`，schema 6；新增 spectrum regions、
  low-rank oracle、response-active projector/cross-covariance。
- tests：`tests/test_evaluate_dfpt.py`；覆盖三段谱分割、rank-6 exact、rank-1
  nontrivial、projector 对列混合不变、相同子空间但错误 sign 的 cross-covariance
  失败。
- exposure ledger：`src/piezojet/train.py` 的最终 `summary["exposure"]`。
- 新 evaluator 产物：
  `outputs/direct_u_multistream_smoke_v1/dfpt_test_schema6_alignment_v2.json` 与同名
  CSV。它是 seed-42 one-pass smoke，不是性能表。

## 10. 文献事实与 PiezoJet 推断的边界

- Moore--Penrose 逆的投影恒等式来自 generalized inverse 的标准定义；本项目
  detached-chart 退化结论则是把该恒等式代入 PiezoJet chart 后的直接推导。
  参考：R. Penrose, *A generalized inverse for matrices*, Proc. Cambridge
  Philos. Soc. 51, 406--413 (1955)。
- DFPT 中 force constants、Born effective charges 与介电响应的定义是文献事实；
  v7 BEC 轴转换、strict completion 和当前 closure 数字是本项目的数据审计结果。
  参考：X. Gonze and C. Lee, *Dynamical matrices, Born effective charges,
  dielectric permittivity tensors, and interatomic force constants from
  density-functional perturbation theory*, Phys. Rev. B 55, 10355 (1997)。
- SVD 最优低秩逼近是线性代数事实；rank-4 的 19.81% response error、projector
  overlap 0.6471 以及 480/480 谱区统计均是本项目 frozen test20 one-pass 产物，
  不应外推为材料体系普遍定律。
