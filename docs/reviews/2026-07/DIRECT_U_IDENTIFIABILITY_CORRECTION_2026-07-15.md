# Direct-U 可辨识性修正实施报告（2026-07-15）

## 结论

本轮不是调 loss，而是修正了旧 observable/pInv 路径的可辨识性和计算图。
维护中的离子响应现在由独立的原子级内部位移响应

\[
U_\eta=\partial u/\partial\eta
\]

给出：

\[
e^{\mathrm{ion}}_U=\frac{c_e}{\Omega}Z^{*\mathsf T}U_\eta.
\]

生产路径不再包含 Moore--Penrose/ridge lift、active/null 投影、detached
predicted-factor chart 或 predicted-spectrum `auto` 分支。`Phi/Lambda` 响应仍
保留为独立物理诊断，不再与 direct ionic 在前向中代数恒等。

## 为什么旧实现不正确

旧路径先预测宏观离子张量 `y`，再用

\[
\Lambda=A^+y+(I-A^+A)R,\qquad
A=Z^{*\mathsf T}\mathcal D_\delta(\Phi)
\]

构造 `Lambda`。当 `A` 满行秩时，重新传播得到 `AA^+y=y`。因此：

1. factorized ionic 前向并没有检验 `Z* / Phi / Lambda` 是否正确；
2. detach `A` 后，前向仍是恒等，反向却包含与真实函数导数不一致的
   straight-through/ghost gradient；
3. 宏观 18 个分量无法识别 `3N x 6` 的完整 `Lambda`，null component 的
   学习只能依赖真正的原子级标签。

这一路径已经从 `model.py` 删除，而不是改名或默认关闭。

## 新模型

### 1. 独立 U_eta head

`CartesianInternalStrainHead` 的独立实例预测每个原子的
`[3,3,3]` 对称应变响应，并施加平移和为零。主离子预测只依赖预测 BEC
和预测 `U_eta`。

同一模型仍预测：

- `Z*`：Born effective charge；
- `Phi`：原子坐标 Hessian；
- `Lambda`：strain-force coupling；
- `factorized_ionic_piezo`：`Z*/Phi/Lambda` 的独立诊断响应。

`direct U_eta` 与 `factorized Phi/Lambda` 初始时不相等。梯度测试确认：

- direct ionic loss 进入 `U_eta` head 与 BEC head；
- 它不进入 `energy_factors`；
- factorized diagnostic 的梯度进入 BEC 与 energy-factor heads。

### 2. 三类 U_eta 监督

在有 OUTCAR branch 标签的材料上，用真实 BEC 隔离 `U_eta`：

\[
\mathcal L_{U,\mathrm{macro}}
=\rho\left(\frac{c_e}{\Omega}Z^{*\mathsf T}_{\rm true}
\widehat U_\eta-e^{\mathrm{ion}}_{\rm OUTCAR}\right).
\]

在 strict-complete 材料上构造：

\[
U_\eta^\star=\mathcal D_\delta(\Phi_{\rm true})\Lambda_{\rm true}.
\]

同时加入不通过预测逆矩阵的 normal equation：

\[
(\widehat\Phi^2+\delta^2I)\widehat U_\eta
=\widehat\Phi\widehat\Lambda.
\]

direct/factorized consistency 只在 strict、通过同一 regularized closure 的
标签上执行。目标端 detach，避免 consistency 反向把 direct head 变成旧式
因子 chart。

### 3. macro / physical 双塔

总张量不能识别 electronic/ionic 分配。现在：

- macro tower 只学习 GMTNet total；
- physical tower 学习 same-OUTCAR electronic、ionic、branch sum；
- macro pass 只构造 macro encoder/head，不构造物理因子图；
- 单元测试证明 macro total loss 对 physical encoder、`Phi/Lambda/U_eta`
  没有梯度。

## 算子修正与审计

默认算子唯一为：

\[
\mathcal D_\delta(\Phi)
=\operatorname{Re}(\Phi+i\delta I)^{-1}
=\Phi(\Phi^2+\delta^2I)^{-1}.
\]

它在 `3N-3` optical basis 中用 complex solve 实施，不物化 dense inverse，
不使用 normal-equation solve 计算前向，因此不平方条件数。`auto` 被删除；
exact 仅能显式请求，并只用于真实 DFPT 稳定子集诊断。

新增审计覆盖：

- actual solve 与显式 eig filter 一致；
- optical basis 任意正交旋转不改变结果；
- 重复、近重复和零特征值无 NaN；
- autograd VJP 与 central finite difference 相对误差小于 `1e-4`；
- float64 basis forward relative error 小于 `1e-10`。

审计发现并修复了一个实际精度 bug：Helmert 分母原先先对整数 tensor
执行 `sqrt`，得到 float32 后才转 float64，导致基底正交误差约
`1.4e-7`。现在索引先转 solve dtype，再计算平方根。

## 标签空间和 loss

OUTCAR total 与 ionic 分别经过同一 point-group Reynolds projection，
electronic 定义为 projected total 减 projected ionic。非零极性材料测试
验证三者在 `2e-7` 容差内闭合。

原先 componentwise SmoothL1 会随坐标旋转改变。现在 dielectric、BEC、
macro piezo、force constant、完整/打印 Lambda、U_eta 和一致性损失，都先
在完整 Cartesian tensor 上形成 Frobenius norm，再做 pseudo-Huber。

样本粒度显式定义：

- dielectric/piezo：每材料；
- BEC：每原子；
- ragged `Phi/Lambda/U_eta`：每材料；
- printed blocks：每观测 block。

共同随机旋转测试在 float64 下通过 `1e-11` 相对容差。

## Exposure-matched 协议

旧 full-corpus 实验是 fixed optimizer updates，不能直接解释为数据规模
实验。新协议的单位是完整 stream passes：

1. factor stage：完整 branch pass + 完整 strict pass；
2. joint stage：完整 macro pass + branch pass + strict pass；
3. strict-only 的 full-Lambda、U target、normal equation 不在 branch stream
   重复；
4. matched direct control 使用相同 macro passes、split、seed、结构预训练
   checkpoint 和 validation-loss selection。

注册 grid：passes `1,5,10,20`，seeds `42,7,1729`。

当前训练 split：

- macro train：4,961；
- branch train：580（610 archives 中扣除 frozen val/test）；
- strict train：149；
- validation/test：固定 10/20。

## Evaluator schema 6

新版 evaluator 分开报告：

- `direct_pred_z_pred_u_regularized`；
- `factorized_pred_z_pred_phi_pred_lambda_regularized`；
- true/pred `Z* / Phi / Lambda` substitution grid；
- strict `U_eta*` 节点级误差；
- true/pred 最低三条 optical eigenvalues；
- true/predicted optical spectrum 的 `<delta`、`[delta,3delta)`、`>=3delta`
  三段 count/fraction；
- `delta/10, delta, 10delta` amplitude sensitivity；
- stable / soft-positive / unstable 分层；
- exact 只在 true-DFPT stable stratum；
- true `U_eta` 的 rank-1/2/4/6 SVD oracle，明确 RHS matrix rank 不等于
  physical phonon-mode count；
- predicted/true `U_eta` 的 column-projector、principal-angle 与 true-BEC
  `Z*^T U` cross-covariance alignment。

## 一次完整 smoke

输出目录：`outputs/direct_u_multistream_smoke_v1/`。

运行包括一个 factor branch/strict pass 和一个 joint macro/branch/strict
pass，使用 RTX 4060 Ti，端到端训练约 419 秒，随后 evaluator test20 约
19 秒。

结果：

| 指标 | one-pass smoke |
| --- | ---: |
| Total TRS | -0.00405 |
| direct-U ionic macro cosine | -0.01505 |
| direct-U ionic amplitude ratio | 0.00537 |
| direct-U ionic MAE skill vs zero | -0.00043 |
| factorized ionic macro cosine | -0.04038 |
| stable / soft-positive / unstable | 14 / 0 / 6 |

这些数字只证明实现可以运行；一个 pass 仍明显振幅塌缩。

schema-6 二次审查还显示：true/predicted 共各 480 个 optical modes 全部位于
`|lambda| >= 3 delta`，因而没有当前 checkpoint 通过预测极软模式压低响应的
证据。rank-4 true-`U_eta` oracle 保留 `0.9598` 的位移奇异能量，但 true-BEC
response 相对误差仍为 `0.1981`；rank-6 才数值精确。true-BEC 下
`Z*^T U_pred` 对 `Z*^T U_true` 的平均 cosine 为 `0.1153`、幅度比为
`0.0563`。这些仍是 one-pass 诊断，不是性能结论。

strict true-factor substitution 给出：

| 组合 | ionic macro cosine | amplitude ratio | MAE skill |
| --- | ---: | ---: | ---: |
| true Z / true Phi / true Lambda | 0.7947 | 0.6908 | 0.9812 |
| pred Z / true Phi / true Lambda | -0.0722 | 0.3165 | -0.0687 |
| true Z / pred Phi / true Lambda | -0.4735 | 0.6609 | -0.2328 |
| pred Z / pred Phi / true Lambda | -0.0146 | 0.1100 | -0.0090 |

true-factor macro cosine受到近零材料逐材料 cosine 的影响；更适合检查闭合
的 component-micro cosine 是 `0.99997`，component MAE 是
`0.00452 C/m^2`。因此数据与声明算子可以闭合，one-pass 的主要误差来自
预测 BEC/Phi/U_eta，而不是旧 lift 的 rank 或 gradient policy。

## 清理结果

从维护代码删除：

- `observable_internal_strain` pInv/ridge lift；
- predicted-spectrum `auto` operator；
- sketch/hybrid training objective；
- mode-aware training branch；
- protocol A--G 可执行训练器；
- production config 中的 observable/ridge/detach 开关。

通用梯度冲突测量保留在 `gradient_diagnostics.py`，但生产 trainer 不导入
它。历史实验结果保留在既有 `outputs/`，用于审计而不是复跑当前模型。

## 尚未完成的性能结论

尚未运行完 1/5/10/20 passes × 3 seeds，因此目前不能回答 direct-U 是否
提升最终泛化，也不能把一次 smoke 与旧 fixed-update 表直接比较。正确的
下一步是执行 `scripts/run_exposure_matched_replay.ps1`，在每个 pass 点同时
比较 physical model 与 matched direct control，并只使用 validation 选择
checkpoint。
