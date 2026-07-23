# PiezoJet electronic generator：长期实验与方向选择规划

## 1. 当前证据与主判断

截至 2026-07-23，development-only、fold0/seed42/N=800 的最佳 A0-PM
checkpoint 位于 update 800：总 stabilized score 为 1.35458，其中 electronic、
BEC、dielectric 分别为 0.50531、0.39123、0.45804。它显著优于旧 A0-PM
的 1.58657，但只比 BEC-only response-aware 初始化的 1.35559 好约 0.001。
因此，response-aware 初始化是有效的训练改进，却没有单独解决 electronic
泛化瓶颈。

128-material scale--shape 诊断进一步表明：全局 amplitude calibration 只把
active relative error 从 0.9281 改到 0.9212，active cosine 保持 0.3292；按
irrep 缩放反而把 cosine 降到 0.2723。两个 `l=1` copy 的 calibration scalar
分别为 3.963 和 0.114，且它们在 audit slice 上的 active cosine 为 -0.040
和 -0.183。即使使用不可部署的 per-material true-norm oracle，cosine 仍为
0.3292。

据此，未来一阶段的主假设应是：

> electronic 的主要剩余误差是等变形状与 multiplicity-space 方向问题，尤其
> 是两个等价 `l=1o` copy 的混合/分工，而不是一个统一的振幅塌缩。

现阶段不应同时更换 backbone、PCGrad、长程模块和损失；否则无法判断哪项
改变真正有效。A0-PM 的三个任务继续参数隔离，BEC/dielectric 作为固定对照。

## 2. 总体决策树

```text
只读 l=1 诊断
  |
  +-- 全局 2x2 multiplicity mixer 明显改善 --> 实现等变 l=1 mixer head
  |      |
  |      +-- same-ID capacity 通过 --> N=200 --> N=800 --> 3 seeds
  |      +-- same-ID capacity 失败 --> 重写 l=1 readout，不进 development
  |
  +-- 2x2 mixer 无改善 --> 检查表示容量与公式 OOD
         |
         +-- same-ID 好、formula-OOD 差 --> 条件化 mixer / chemistry adapter
         +-- same-ID 也差 --> l=1 head/局域-全局耦合表示不足
         +-- l=2/l=3 同时差 --> 再考虑 backbone 或长程特征
```

## 3. Stage 0：冻结协议与可复用资产

目标是保证后续差异只来自候选本身。

- 冻结当前 A0-PM update-800 checkpoint、N=800 manifest、fold0 development
  IDs、graph schema/cache key、stabilized metric 和 guardrails。
- 结构、BEC response-aware、electronic response-aware checkpoint 全部复用；
  语义、width、fold、material-ID hash 不匹配时直接报错，不 fallback。
- frozen validation10/test20 在所有方向选择阶段保持未读。
- 每个新 cohort 保存命令、commit、数据 hash、所有 evaluation checkpoints、
  training curve、per-material/irrep 指标和失败原因。

通过标准：能够从同一 checkpoint 重放完全相同的 128-material baseline 指标。
预计成本：小于 10 分钟，无训练。

## 4. Stage 1：`l=1` multiplicity-space 对抗诊断

### 4.1 全局 2x2 mixer oracle

两个 `l=1o` copy 是同一 O(3) irrep 的两个 multiplicity。保持等变性时，可以
在 multiplicity 维施加任意共享的 2x2 标量矩阵：

\[
\begin{bmatrix}\hat y'_{1a}\\\hat y'_{1b}\end{bmatrix}
=M
\begin{bmatrix}\hat y_{1a}\\\hat y_{1b}\end{bmatrix},
\qquad M\in\mathbb R^{2\times2},
\]

其中同一个 `M` 作用于三个磁量子分量。先在 calibration64 上用最小二乘拟合
`M`，再在完全分离的 audit64 上评估；同时报告 unconstrained、orthogonal
polar factor 和 diagonal-only 三个版本。这个实验比独立 scalar 更能检验
“两个 copy 被错误混合”的假设，而且不会破坏 O(3) 等变性。

通过门槛：audit64 electronic active cosine 至少提高 0.10，且 stabilized
relative error 至少降低 0.03。若仅 calibration 改善而 audit 不改善，判定为
不可泛化的后处理，不进入模型。

### 4.2 分层与稳定性检查

在完整 development988 上只读报告：

- 两个 `l=1` copy、`l=2`、`l=3` 的 target norm、prediction norm、cosine、
  relative error；
- crystal system、原子数、公式新颖度、target-response bin、稳定/不稳定谱分层；
- 两个 `l=1` copy 的 2x2 cross-covariance、condition number 和 bootstrap
  interval；
- 旋转、原子置换和 batch permutation 下 mixer 前后的不变性。

近零 target 必须单独报告 absolute leakage；不得用 raw relative error做选择。

## 5. Stage 2：表示能力还是公式外推

只对通过 Stage 1 数学诊断的候选运行容量阶梯。

### 5.1 same-ID electronic capacity

使用固定 samples8/32/128，关闭其他任务、augmentation 和 dropout，分别比较：

- 当前 electronic tower；
- 当前 tower + 2x2 `l=1` mixer；
- 必要时独立 `l=1` readout candidate。

每个点报告总体与各 irrep 的 relative error、cosine、amplitude、train time、
peak memory。samples32 门槛：electronic relative error不高于 0.10、总体 cosine
不低于 0.97、两个 active `l=1` copy cosine 均不低于 0.90。未通过时不允许
运行 formula-disjoint development。

### 5.2 frozen synthetic-teacher

用候选自身生成标签，检查 1/8/32 是否可被同一模型类重新拟合。若 synthetic
teacher 也失败，属于实现/优化问题；若 synthetic 通过而真实 same-ID 失败，
属于真实目标不在当前 model class 或训练条件不良。

## 6. Stage 3：最小架构候选顺序

候选严格串行，每次只改变一个因素。

### Candidate M1：全局等变 `l=1` multiplicity mixer

- 在 electronic readout 的两个 `l=1o` copy 后添加一个 2x2 线性层；
- 初始化为单位阵，保持初始预测与 checkpoint 完全相同；
- 不改变 `l=2/l=3`、BEC、dielectric 或 encoder；
- 先仅训练 mixer，再允许 electronic head 联合微调。

这是第一优先候选：参数极少、数学上精确等变、直接对应当前诊断。

### Candidate M2：结构条件化 multiplicity mixer

只有 M1 在 calibration/audit 间表现出材料依赖时才启用。由 invariant global
features 输出接近单位阵的 2x2 residual mixer，并用谱范数或 Frobenius penalty
限制偏离。禁止每个样本使用 target norm，也禁止跨不同 `l` 混合。

### Candidate S1：受限 scale--shape head

只有方向明显改善但 amplitude 仍系统偏低时才启用。每个 irrep 预测

\[
\hat y_l=\exp(s_l)\,\frac{q_l}{\lVert q_l\rVert+\epsilon},
\]

其中 `s_l` 和 `q_l` 均只依赖结构。active targets 使用 angular + log-amplitude
损失；近零 targets 使用 absolute leakage 损失。scale、shape 和 zero leakage
必须分别报告。当前结果不支持直接跳到 S1。

### Candidate H1：独立 `l=1` response path

若 M1/M2 的 same-ID capacity 仍失败，为两个 `l=1` multiplicity 建立独立的
局域/全局 readout 或 factor-specific adapter，同时保留统一 electronic tower。
它解决表示容量，不重新共享 A0 的任务参数。

### 延后方向

- PCGrad/GradNorm：A0 的任务参数本来分离，不能修复 electronic 内部 shape。
- 更大 backbone/lmax：只有 `l=2/l=3` 与 `l=1` 同时显示容量不足才合理。
- 新长程模块：只有误差与晶胞尺寸/离子性/长程分层显著相关时启动。
- 非线性 polarization tower：零点 Jacobian 标签无法识别额外非线性；需要
  finite-perturbation/field 标签后才有可证伪意义。
- 简单后处理 scalar：已被当前 audit 否决为主要解法。

## 7. Stage 4：快速 inductive gate

每个通过 same-ID 的候选按以下顺序运行：

1. N=200、fold0、seed42、最多 400 updates；每 25--50 updates 完整 development
   评估，guardrail-aware early stopping。
2. N=800、fold0、seed42、最多 1000 updates；复用全部兼容 checkpoint。
3. 只有 N=800 相对当前 1.35458 明显改善，才进入三 seed。

单-seed 晋级门槛：

- 总 development score 至少下降 0.03；
- electronic stabilized relative 至少下降 0.04；
- electronic active cosine 至少提高 0.10；
- active amplitude 保持 0.2--2.0，且不能靠极端 norm 放大换取 cosine；
- BEC 和 dielectric 各自退化不超过 0.02；A0 参数隔离下理论上应完全不变；
- 无 equivariance、zero-target leakage、内存或吞吐 guardrail 失败。

没有达到门槛的候选保留为负结果，不继续增加 updates 来“磨”出差异。

## 8. Stage 5：稳健性、规模与最终晋级

### 三 seed

对单-seed 胜者运行 seeds 42/7/1729。要求平均 improvement 仍为正，且效应
不小于 seed 间样本标准差；同时报告配对每材料差值，而不只报告均值。

### 五个 development folds

三 seed 通过后，再运行至少五 fold 的单 seed 或资源允许时的 3x5 设计。
重点检查公式 OOD、晶系、原子数和 response magnitude 分层是否稳定。若只在
fold0 有效，判定为 fold-specific，不进入论文主结果。

### 扩充 response pretraining

架构通过 N=800 后，才把 electronic response-aware pretraining 从 N=800 扩到
fold-train 的全部可用 formula-safe response labels。它是数据规模实验，必须与
固定架构、固定 updates/exposures 的 N=800 control 成对比较。

### frozen panels

候选、超参数、checkpoint rule 和 seeds 全部预注册后，才读取 validation10；
test20 只供最终一次报告。任何失败不得返回 development 继续改同一候选后再次
查看 test20。

## 9. 数据方向

短期不把主要资源投入重新下载 DFPT。现有 4,995 parsed payloads 的 BEC、
electronic piezo、dielectric 和 Phi 字段覆盖足以做 electronic generator 判断，
当前也没有 convention bug 的证据。

真正有价值的数据工作是：

- 对 4,939 electrostatic formula-safe records 计算 response norm、irrep energy、
  crystal system、atom count、composition novelty 分层；
- 检查 N=800 subset 是否低估两个 `l=1` copy 的高响应材料；
- 若存在明显覆盖缺口，重建一个只改变 sampling 的匹配 N=800 control；
- 未来若能获得有限位移/有限应变 polarization 数据，再单独验证 nonlinear tower。

不得通过放松 strict Lambda completion 来增加样本，也不得把 MP 标签混入
JARVIS-only benchmark。

## 10. 计算与工程预算

- Tier 0，只读 oracle/分层：1--10 分钟，CPU 或单 GPU。
- Tier 1，samples8/32/128 capacity：约 1--4 GPU 小时/候选。
- Tier 2，N=200 单 seed：约 1--2 GPU 小时。
- Tier 3，N=800 单 seed：当前约 4--6 GPU 小时，可 early stop。
- Tier 4，三 seed：三张 4090 并行，约 5--8 小时 wall time。
- Tier 5，五 fold：只对一个最终候选执行，约 25--80 GPU 小时，取决于 seeds。

训练代码继续使用持久 DataLoader、预取、canonical graph cache、immutable
evaluation checkpoint 和断点续训。先 profile 再优化；不以改变数据顺序或
selection rule 的方式换取速度。

## 11. 论文结果策略

- 当前可以报告：A0-PM 参数匹配优于 shared A1/A1.6；response-aware 初始化
  显著改善旧 A0-PM；简单 amplitude calibration 不能解释剩余 electronic
  误差。
- `l=1` mixer 在三 seed/多 fold 前只写入 appendix 作为 development hypothesis。
- 同时保留负结果：per-irrep scalar、oracle norm、A1/A1.6、零门控 A1.5。
- 不把同 ID capacity 结果写成 inductive performance，也不因总张量未胜 direct
  baseline 而夸大 physical generator 的总体优势。

## 12. 最近三步的实际执行顺序

1. 实现只读 global 2x2 `l=1` multiplicity mixer oracle，并在现有 calibration64/
   audit64 与完整 development988 上运行。
2. 若通过门槛，实现 identity-initialized M1，并跑 samples8/32/128 same-ID
   capacity；否则直接进入分层 OOD/表示诊断。
3. 只有 M1 capacity 通过时，跑 N=200/fold0/seed42 快速 inductive gate。

这三步完成前，不启动新的大规模预训练、不启用 PCGrad、不扩 backbone，也不
读取 frozen validation10/test20。

## 13. 执行记录（2026-07-23 更新）

### 13.1 三折 A0-PM 基线

三折单 seed（seed 42、N=800、development-only）按同一 stabilized selection
协议运行。fold0 沿用已完成且 provenance 兼容的旧 cohort；fold1、fold2 使用
同一参数匹配架构、response-aware BEC/electronic 初始化和断点协议。

| fold | 状态 | selected update | stabilized score | electronic | BEC | dielectric |
|---|---|---:|---:|---:|---:|---:|
| 0 | complete（兼容 cohort） | 800 | 1.35458 | 0.50531 | 0.39123 | 0.45804 |
| 1 | complete_early_stopped | 700 | 1.51672 | 0.53235 | 0.38265 | 0.60172 |
| 2 | complete | 1350 | 1.34754 | 0.49539 | 0.38071 | 0.47144 |

fold1 在 update 700 后满足四次无改善的早停条件；fold2 在 update 1350 取得
最佳 checkpoint，并于 update 1500 正常完成（最终 guardrails 全通过，selected
electronic active cosine 0.48425、BEC nonzero cosine 0.88517、active amplitude
ratio 0.42431）。三折的意义是确认 A0-PM 的
fold 稳定性，不是冻结测试集结果；validation10/test20 仍未读取。

三折 selected score 的均值/样本标准差为 1.40628/0.09571；electronic、BEC、
dielectric 分项均值分别为 0.51102、0.38487、0.51040。fold1 的 dielectric
分项明显较高，说明晶体/公式 OOD 仍有 fold variance；这不是理由去读取 frozen
panel 或改变 selection rule。

### 13.2 M1 global `l=1` mixer oracle 判定

`outputs/electronic_mixer_oracle_fold0_seed42_v1/` 已完成 calibration/audit
和完整 development988 的只读诊断。unconstrained 2×2 mixer 与
orthogonal-polar mixer 均未达到预注册门槛（audit active cosine 至少 +0.10、
relative error 至少 -0.03）；完整 development 上 baseline / unconstrained /
orthogonal-polar 的 active relative error 为 0.87115 / 1.00273 / 0.87138，
cosine 为 0.43404 / 0.41004 / 0.43913。故 M1 在 oracle 阶段拒绝，不运行 same-ID
capacity，不进入 production，也不启动 M2 的长训练。

这一负结果把后续搜索从“全局振幅/固定 mixer 后处理”收敛到表示与公式 OOD
分层诊断。下一最小动作是复用现有 checkpoint 做 per-irrep、晶系、原子数、
response-scale 和 formula-novelty 分层；只有分层证据支持材料条件化，才考虑
结构条件 mixer，否则优先评估独立 `l=1` readout 的小容量候选。

### 13.3 分层诊断的首个结果

新增只读工具 `piezojet.electrostatic_stratified_diagnostic`，并在 fold2
update 1200 checkpoint 上运行。它验证 checkpoint 的 development ID 集合与
fold manifest 完全一致，并报告 `frozen_validation_test_labels_read=false`。
官方 dft_3d 元数据覆盖 977/987 个 development ID，仅用于晶系/空间群/公式
描述，不参与任何标签或 checkpoint 选择。

在这个中途 checkpoint 上，electronic active panel 为 436 个材料，active
relative error 0.86977、cosine 0.46721、amplitude 0.39622。按 response norm
分层时，目标范数 `[0.5,1)` 的 cosine 仅 0.345、`[0.1,0.5)` 为 0.427，而
`[1,inf)` 为 0.510；这说明误差不能由一个全局 scalar 振幅解释。按晶系，
tetragonal 的 active cosine 0.610，而 orthorhombic/trigonal 分别为 0.379/0.403；
按原子数也没有单调关系（3--8 atoms 为 0.555，9--16 为 0.434，17--32 为
0.469）。这些是 development-only 的方向证据，不能外推到 frozen panel。

因此下一候选暂不启用结构条件 mixer；优先完成 fold2 后，对选定 checkpoint
补齐相同分层与公式新颖度报告，再决定是否值得做一个独立 `l=1` readout 的
samples8/32 capacity。该候选若不能先通过 same-ID gate，不进入 N=200 或
更大规模训练。

对两个 `l=1` copy 做 per-material 2×2 拟合的 spread 诊断显示 audit 中
global-map 偏差中位数约 9.85、P90 约 65.38，但单材料只有三个 vector
components，拟合矩阵条件数中位数约 4.6×10^5，因而该 oracle 过度欠定，不能
直接证明应部署材料条件 mixer。它只支持“固定全局 mixer 不足”的结论；下一
候选必须先用同 ID capacity 证伪，而不能把这个欠定 oracle 当作性能结果。
### 13.4 Independent `l=1` readout capacity decision (2026-07-23)

After the global mixer oracle failed, we tested the smallest readout-only candidate: retain the standard `l=2/l=3` readouts and replace the two `l=1` copies with independent readouts. Candidate and `a0_parameter_matched_irreps` baseline used the same fold-0 train-only structure checkpoint, identical 8/32 material manifests, optimizer, and batch settings. No development or frozen-panel labels were read.

The 1-epoch samples8 smoke gave baseline active electronic relative error/cosine 0.8510/0.7822 versus 0.9798/-0.0817 for the candidate. The completed 20-epoch samples32 same-ID capacity result was:

| architecture | best epoch | active relative error | active cosine | amplitude ratio |
|---|---:|---:|---:|---:|
| A0 parameter-matched baseline | 20 | 0.68937 | 0.60649 | 0.43342 |
| independent `l=1` readout | 19 | 0.82108 | 0.55340 | 0.23192 |

The candidate was worse at epochs 1/5/10/15/20 and did not meet the preregistered error (-0.04) or cosine (+0.10) gate. It therefore does not enter N=200, N=800, or the final three-seed study. Full outputs, logs, checkpoints, and the train-only manifest are retained under `outputs/electrostatic_l1_capacity_v1/`.

This rejects “extra independent `l=1` readout capacity” as the next explanation; it does not indicate a data or tensor-convention failure. The next action is a read-only per-irrep/response-scale audit. A scale--shape candidate is authorized only if it shows stable direction with amplitude collapse, and any new candidate must pass samples8/32 same-ID capacity before inductive development.
