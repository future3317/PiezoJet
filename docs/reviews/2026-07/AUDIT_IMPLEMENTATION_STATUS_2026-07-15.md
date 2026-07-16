# 外部技术审查实施状态（2026-07-15）

本文按 [`详细审核和调查.md`](详细审核和调查.md) 的可证伪假设核对当前代码。
“完成”表示实现、测试和最小证据齐全，不表示预测性能已经改善。JARVIS-only
标签、strict completion 门槛和冻结的 149/10/20 train/validation/test 面板均未改变。

## 已完成的必须修复项

| 审查项 | 状态 | 当前实现与证据 |
| --- | --- | --- |
| P0-1：删除同一预测映射的 pInv/ridge chart | 完成 | 生产模型不再从宏观 ionic tensor lift `Lambda`，也没有 active/null projector、detach chart、SVD cutoff 或 straight-through 梯度。主离子路径为独立原子级 `e_ion=(c_e/Omega) Z*^T U_eta`。完整 `Phi/Lambda` 只形成独立 factorized diagnostic。 |
| P0-2：按数据暴露量训练 | 实现完成，完整实验待运行 | `train.py` 使用 macro/branch/strict 三条流；factor 阶段遍历 branch+strict，joint 阶段遍历 macro+branch+strict，并记录各流 passes、examples、unique coverage、optimizer updates，以及每类标签实际进入梯度的 update 数。`scripts/run_exposure_matched_replay.ps1` 注册 1/5/10/20 passes × seeds 42/7/1729，同时运行 matched direct control。 |
| P0-3：光学基、算子和 VJP | 完成 | 生产只允许连续 regularized operator；predicted-spectrum `auto` 分支已删除，exact 只用于 true-DFPT stable diagnostic。Helmert optical basis 的 dtype 顺序已修复。随机基变换、重复/零本征值、complex solve 对显式谱滤波及 VJP 对中心有限差分均有测试；VJP 相对误差阈值为 `1e-4`。 |
| P0-4：旋转不变 loss 与 branch target space | 完成 | tensor loss 先计算完整 Cartesian Frobenius norm，再做 pseudo-Huber reduction；不再逐 Cartesian component 做 robust reduction。OUTCAR total 与 ionic 先经过同一 Reynolds projection，electronic 定义为 projected total minus projected ionic，branch closure 有回归测试。 |
| P0-5：factor/operator/subspace diagnostics | 完成 | schema-6 DFPT evaluator 输出 direct-U、factorized diagnostic、true/pred factor substitution、`U_eta*` 节点误差、optical spectrum 三段 count/fraction、delta sensitivity、stable/soft-positive/unstable 分层、rank-1/2/4/6 true-`U_eta` oracle，以及 projector/principal-angle/`Z*^T U` cross-covariance。exact 只在 true-stable 子集出现。 |

## 已完成的架构修正

| 审查项 | 状态 | 当前实现与证据 |
| --- | --- | --- |
| P1-1：独立 direct-`U_eta` head | 完成 | 新增平移投影后的 atom-resolved displacement-response head。179 个 strict records 监督 `U_eta*=D_delta(Phi)Lambda`；610 个 same-OUTCAR branch records 用真实 BEC 监督其宏观作用；normal-equation loss 约束独立预测的 `Phi/Lambda/U_eta`，不通过 inverse 反传。 |
| P1-4：physical/macro 双塔 | 完成 | 4,961 个 GMTNet total labels 只更新独立 macro tower；same-OUTCAR electronic/ionic 与 strict factor labels 更新 physical tower。`predict_macro_total()` 不构造 physical graph。total-only 梯度无法重写 `Z*`、`Phi`、`Lambda` 或 `U_eta`。 |
| Partial-label masking | 完成 | piezo/dielectric mask 按材料，BEC mask 按原子，ragged `Phi/Lambda/U_eta` mask 按材料；strict-only consistency 不再在 branch stream 重复。 |
| 生产架构收敛 | 完成 | `PiezoJet` 和 `model_from_config` 只接受 `energy_learned_strain`、isotropic background 和 regularized production operator。legacy/local-star、anisotropic candidate、mode-aware、sketch、A--G trainer、pInv/ridge 与 compatibility fallback 均已删除。 |

## 数据与来源审计

| 审查项 | 状态 | 当前实现与证据 |
| --- | --- | --- |
| BEC 轴与 Lambda 符号 | 完成 | cache 保留 VASP source `Z[i,j]=dP_i/du_j`，入口只转置一次为内部 coordinate-row convention。OUTCAR printed internal strain 保持 `dF/deta=Lambda`，不作全局符号翻转。相同 strict gates 下得到 179 个 closure-complete labels。 |
| GMTNet total 对 same-ID OUTCAR total | 完成 | 610/610 raw totals 经一次共同转换后精确一致。仅全局训练中的 JVASP-42995 和 JVASP-28862 在 Reynolds-projected branch target 上触发双阈值冲突，因此只 mask 其 DFPT macro branch supervision；二者不在冻结 149/10/20 面板。 |
| completion condition/provenance | 完成 | schema-4 provenance cache 记录输入 digest、parser identity、转换、units、tensor checksum、`sigma_min(MB)` 与 condition number；v8 保持 v7 的 179 个 accepted IDs，不改变 gate。 |
| 外部 parser/geometry oracle | 完成但边界有限 | 10 个 raw archive 通过 pymatgen/Phonopy 文本复算；ASE primitive-neighbor-list 与 symfc projection 为只读交叉检查。它们验证 parser/几何局部环节，不重写 source arrays，也不声称已经解决 NAC/LO--TO 边界条件。 |

## 当前最小实验结果

`outputs/direct_u_multistream_smoke_v1/` 是 seed-42、每条流一个完整 pass 的实现
smoke，不是性能表：

- total TRS：`-0.00405`；
- direct-U ionic material-macro cosine：`-0.01505`；
- direct-U ionic amplitude ratio：`0.00537`；
- direct-U ionic MAE skill：`-0.00043`；
- factorized diagnostic ionic macro cosine：`-0.04038`；
- true `Z*/Phi/Lambda` closure：component MAE `0.00452 C/m^2`，component-micro
  cosine `0.99997`；
- true-DFPT stability strata：stable/soft-positive/unstable = `14/0/6`。
- true/predicted optical spectrum：各 `0/0/480` modes 位于
  `<delta / [delta,3delta) / >=3delta`；
- rank-4 true-`U_eta` oracle：位移能量保留率 `0.9598`，true-BEC response
  相对误差 `0.1981`；
- true-BEC cross-covariance `Z*^T U_pred`：cosine `0.1153`，幅度比
  `0.0563`。

该结果证明新计算图、数据流和 evaluator 闭合，但尚不支持精度改进。用 one-pass
预测的 `Z*` 或 `Phi` 替换真因子会明显破坏 strict closure，预测 `U_eta` 本身也接近
零技能；当前瓶颈因此是 learned factor/displacement quality，而不是被删除的 chart。

## 有意未实现或仍待证伪的方向

- P1-2 mass weighting、relative regularization 和 scale-shape 参数化：需要先完成
  exposure replay，再以完全 matched 的 validation-only ablation 判断；当前不加入
  未验证的生产回退分支。
- P1-3 reduced response-subspace：ordinary SVD oracle 已完成；rank-4 response
  误差仍为 19.81%，rank-6 的精确重建只是六个 RHS 的线性代数事实。`r=6` 输出
  `18N+36` 个量，不压缩 direct-`U` 的 `18N` 个量且增加 latent gauge，因此不实施
  rank-6 head；只有 response-weighted `r<6` oracle 达标后才重议压缩架构。
- P1-5 新 strain-force/relaxed-displacement 数据：需要新的 DFPT/VASP 计算权限和
  formula-diverse acquisition protocol，不能由现有 total labels 伪造。
- NAC/LO--TO、SSCHA/TDEP 和概率潜变量属于后续边界条件或数据扩展，不在当前
  homogeneous zero-temperature source定义上静默启用。

## 下一有效实验

运行 `scripts/run_exposure_matched_replay.ps1` 的 1/5/10/20 passes × 3 seeds，按
validation loss 选择 checkpoint，并同时报告 matched direct control。该 replay 是
两个并行实验：（1）149/610 个 physical-label 样本在充分曝光下能否学到
direct-`U_{eta,delta}` ionic response；（2）4,961 条 total-only 标签是否足以训练
独立 macro total predictor。由于两塔梯度隔离，它不检验 total-only 标签是否改善
ionic factors。mass/scale 或 reduced-response ablation 只能依据 validation 或
train-fold 证据决定。
