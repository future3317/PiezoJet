# PiezoJet 独立全库数学与实现审计（2026-07-18）

## 1. 总体 verdict

**结论：部分正确。** 当前生产代码的数学主体在审计与最小修复后与声明的方法一致：

- `Phi/Lambda/C` 是明确二次 factor energy 的系数；`Phi` 与 `Lambda` 不共享旧 `K/S` 强耦合；
- production ionic 路径是独立的 `Z*^T U_{eta,delta}`，factorized resolvent 仅为诊断和严格标签目标；
- macro、physical direct-U、factorized diagnostic 的参数和训练路由确实分离；
- regularized signed resolvent、optical projection、exact-stable diagnostic 和一阶 U/V 系统的符号一致；
- BEC 转置、`Lambda=dF/deta`、engineering shear、Cartesian contraction 和响应单位在被审路径中一致。

但不能给出“完整且已验证有效”的 verdict。审计开始时存在两个会否定旧 benchmark/图构造可信度的阻断性错误：强倾斜晶胞会漏 PBC 邻居，所谓 formula-disjoint split 使用了未约分晶胞计量式。二者已修复，旧 val10/test20 ID 未改变，但新 4944/1595 split 尚未训练。因而当前只能确认**数学与执行结构在已覆盖路径上正确**，不能确认**新协议下的预测有效性**。

本审计未读取 frozen test20 标签，未运行长训练。

## 2. Findings

### 阻断性：已确认错误，已修复

#### B-01 强倾斜晶胞的 PBC 搜索范围不完备

旧实现用最短直接晶格矢量确定每个整数平移轴的统一范围。对于倾斜晶胞，大整数晶格组合可以相消并落入 cutoff；例如 `4*a-b=(0,-0.2,0)`，旧 `[-1,1]^3` 范围会漏掉 `(4,-1,0)`。

修复：`src/piezojet/data.py` 的 `_periodic_edges` 改用 inverse-cell 列范数给每个整数轴建立完备界；graph cache schema 升至 5，防止复用旧不完备图。`tests/test_pbc_graph_ase_oracle.py` 新增强倾斜晶胞与 ASE 全集合 oracle 的严格集合相等测试。

影响：旧 schema-4 图缓存不能作为强倾斜晶胞生产图的正确性证据；schema-5 会强制重建。

#### B-02 历史 formula-disjoint split 存在 reduced-formula 泄漏

旧 `formula()` 直接拼接一个具体晶胞中的元素计数，没有除以计数最大公约数。独立复算发现：

- train--val：`AlSb, AsGa, CdS, HNaO`；
- train--test：`AlAs, BeF2, HNaO, SeZn`；
- val--test：`HNaO`。

修复：`src/piezojet/data.py` 的 `formula()` 现在返回约分组成；新 canonical 文件为：

- `strict_completion_benchmark_train_v11_reduced_formula_safe.json`：1595/10/20；
- `full_corpus_multitask_train1595_v2.json`：4944/10/20；
- `strict_train1595_development_folds_v2.json`；
- `electrostatic_development_folds_v2.json`。

独立复算确认新 strict/macro split 的 train--val 和 train--test reduced-formula 交集均为空。冻结的 val/test 仍彼此共享 `HNaO`，所以只能声称 train-vs-held-out 分离，不能声称三方 formula-OOD。

### 高：已确认错误，已修复

#### H-01 checkpoint 未绑定完整 cohort/data 身份

旧 checkpoint 主要依赖路径、stage 名和局部配置；同样大小但不同 ID 的 split、已变化的 manifest 或错误 fold 可能静默进入恢复/评估。

修复：新增 `src/piezojet/checkpoint_provenance.py`，绑定 split 文件 SHA256、各 split ID SHA256、完整 assignment SHA256、canonical manifest SHA256、GMTNet data commit、seed、fold identity 和 split kind。训练恢复、factor/teacher-U、direct baseline、evaluate/infer/DFPT evaluator、same-ID electronic capacity、structural pretraining 和 Stage-A selected checkpoint 均接入校验；缺少 provenance 的旧 checkpoint 被当前路径拒绝。

特别规则：只有明确的 `material_ids_same` 非归纳诊断可在 train/val 间复用 ID，并持久化 `noninductive_same_id=true`。它的 split kind 和 assignment hash 不可能匹配 development/production checkpoint。

#### H-02 evaluate/infer 与默认训练可静默采用无关 global split

旧 evaluator 会在找不到 checkpoint split 时调用全局 split 生成器。默认训练也会先生成 global split，再可能被显式 split 覆盖。这不满足“一个 canonical role、无历史/隐式 fallback”的治理要求。

修复：evaluate/infer/direct evaluator 要求 checkpoint 对应的显式 split 并验证 provenance；默认 production training 使用 canonical `multitask_split`，若既无显式 split 又无 canonical role 则报错，不再生成一个替代 benchmark。material-ID global restriction也必须以 canonical multitask split 为基底。

#### H-03 branch-sum 可被配置重新变成损失

虽然配置默认权重为零，旧 `_epoch()` 仍把 `branch_sum_weight * branch_sum_component` 加入目标。一个配置修改即可重新让 U 补偿 electronic error。

修复：优化和式中删除该项，入口和 `_epoch()` 对任何非零权重直接报错；closure 数值和显式梯度诊断仍保留。

#### H-04 Stage-A 计划与 selected checkpoint 的来源治理不完整

README/AGENTS 指向的旧计划仍含已从 maintained runner 删除的 A2/A3 命令；Stage-A `selected.pt` 只记录 architecture/fold/seed。

修复：保留旧计划作为历史证据，新计划写入 `outputs/electromechanical_jet_fold_adjudication_v2/`，仅含 A0/A1/A1.5；每个 selected checkpoint 绑定 fold train/dev ID、fold 文件、canonical manifest、data commit、seed 和 fold identity。未知架构不再进入一个死的 `model.state.encoder` fallback。

### 中：已确认边界或高风险但尚需实验

#### M-01 微批累计是数学等价，不是 bitwise AdamW 等价

对 A0/A1/A1.5 的两材料 CPU oracle 表明，微批与整逻辑批的参数梯度在 `rtol=2e-5, atol=2e-7` 内一致。第一次 AdamW 更新并不逐位一致，因为不同 kernel/reduction 顺序的近零梯度差异会被 Adam 归一化放大。

这不是目标函数变化；所有候选固定相同 logical batch、microbatch、shuffle seed 和每逻辑批一次 optimizer step 时仍是公平比较。但文档不得声称 bitwise optimizer equivalence。若要研究 microbatch 尺寸本身的效果，必须将它作为单独数值敏感性实验。

#### M-02 A0/A1/A1.5 不是参数量匹配的架构对照

A0 有两个互不共享的 encoder，A1 有一个共享 encoder，A1.5 在共享 encoder 上增加 task adapter。它们共享同一 fold-train-only structural state、数据顺序、更新数、optimizer、weight decay 和 selection rule，但参数量不同。因此该实验只检验“独立/硬共享/软共享”的整体模型类，不应解释为在严格固定容量下仅改变共享方式。

#### M-03 新 v11 split 尚无归纳结果

所有 legacy train1603/4961 validation 数字都含 reduced-formula 泄漏，只能保留为机制诊断。严格 provenance 会拒绝用旧 structural/factor/U checkpoint 初始化新 1595/4944 实验。这是正确的安全行为，但意味着目前没有新 split 上的预测有效性结论。

#### M-04 full O(3)/space-group 正确性仍依赖有限 oracle 覆盖

代码和测试覆盖了旋转、反演奇偶、点群 Reynolds 投影、置换、平移、非正交 PBC、完整等距壳层和 ragged batching；这足以发现多个常见实现错误，但不是对全部 4995 个结构和所有空间群操作的形式证明。新 graph-cache-v5 在全语料上的统计重建仍未运行。

### 低：命名或文档问题

#### L-01 内部类名仍含 `Potential`

公开 `ResponsePotential`/兼容别名已删除，代码中不存在旧 `potential()` scalar-generator API。`AtomCoordinateResponsePotential` 仍是内部 optical response/operator helper 的历史类名；其行为和 docstring 不再声称 macro/direct-U/factorized branches 来自共同微观势能。属于命名债务，不是当前数学 fallback；本轮未做大规模重命名。

#### L-02 历史脚本仍可显式重放旧 split

旧 PowerShell replay 保留以复现实验档案，并可能显式传入 train1603/4961 split。它们不会被 canonical config 自动扫描，且产物 registry 已标注 legacy leakage；但操作者仍需避免把一次显式历史重放登记成当前 formula-OOD 结果。

## 3. 数学定义到真实代码路径

| 数学/物理对象 | 生产或诊断实现 | 审计结论 |
|---|---|---|
| `E_fac=1/2 u^T Phi u-u^T Lambda eta+1/2 eta^T C eta` | `IndependentQuadraticResponseHead`；`AtomCoordinateResponsePotential.internal_quadratic_energy` | 符号和混合导数正确；`Phi`/`Lambda` 独立；`C` 为对称 affine curvature |
| `Phi` acoustic/Hessian symmetry | edge-incidence block assembly；`translation_projector`；optical Helmert basis | 两侧 acoustic sum rule 和三条平移零模由构造/投影保证 |
| independent `U_{eta,delta}` | isolated `displacement_encoder` + `OctupoleGlobalDisplacementResponseHead` | 不读取 `Phi/Lambda/BEC/macro`；不是 unstable reference 上的 stationary `du/deta` |
| production ionic `e_ion=c_e/Omega Z*^T U` | `ionic_piezo_from_displacement_response`; `predict_components` | 无 inverse/pInv/ridge chart；physical total 为 electronic + direct-U ionic |
| factorized diagnostic | `responses` + `apply_optical_operator(Phi,Lambda)` | 与 physical direct-U 分开；按需省略，不作为 production ionic fallback |
| signed resolvent | complex solve `Re[(Phi_o+i delta I)^-1]` | 对负/正 eigenvalue 保留符号，幅值有界；不平方条件数 |
| exact stationary solve | `apply_optical_operator(..., solve_policy="exact")` | 仅当 true optical minimum 大于 stability cutoff；production config 拒绝 exact/auto |
| first-order U/V | `displacement_first_order_block_loss` | 使用 `Phi U-delta V=Lambda`, `Phi V+delta U=0`；未重引入 normal equation |
| macro tower | `macro_encoder` + macro piezo/dielectric/elastic heads | 与 physical/factor encoders 参数隔离；输出明确标记 `isolated_macro_total_tower` |
| BEC convention | `source_born_to_internal` | source `dP_i/du_j` 仅在 ingress 转置一次为内部 coordinate-row convention |
| internal strain | `source_internal_strain_to_internal` | OUTCAR printed `dF/deta` 直接作为 `Lambda`，无全局负号补丁 |
| engineering shear | `tensor_ops.py`, `elastic_dielectric_ops.py` | canonical `(xx,yy,zz,yz,xz,xy)`；strain shear 双倍，piezo/elastic contraction一致 |
| PBC graph | `_periodic_edges` | inverse-cell 完备平移界；完整保留 cutoff/neighbor-budget 边界等距壳层 |

## 4. 已执行检查

| 检查 | 结果 |
|---|---|
| 全量 pytest（全部审计修复后） | 199 passed in 107.55s |
| 定向 provenance/PBC/Stage-A/microbatch | 31 passed |
| Ruff `src tests scripts` | All checks passed |
| experiment registry 生成与 `--check` | 139 cohorts / 2706 artifacts，coverage valid |
| reduced-formula 独立交集复算 | strict 1595/10/20、macro 4944/10/20；train-held-out 无交集，val-test=`HNaO` |
| PBC ASE oracle | 强倾斜反例集合严格相等，包含 `(+/-4,-/+1,0)` |
| 论文编译 | `latexmk -pdf ...` 成功，20 页；第 7--9、13--15、18--20 页目检无裁切/重叠 |
| frozen test20 | 未读取标签、未计算指标 |
| 长训练 | 未启动 |

## 5. 修复内容

1. reciprocal/inverse-cell PBC 完备搜索界与 cache schema 5；
2. reduced-formula 规范化及新 1595/4944 canonical split/folds；
3. checkpoint/data/fold provenance 模块及所有 maintained 恢复/评估入口校验；
4. 显式 evaluator/infer split，去除 production generated-global fallback；
5. branch-sum 永久诊断化；
6. 删除公开旧 `ResponsePotential` 兼容接口，澄清共同能量声明边界；
7. Stage-A selected checkpoint provenance、未知架构 fallback 删除和新 v2 非执行计划；
8. A0/A1/A1.5 微批梯度回归覆盖，并纠正 bitwise AdamW 等价表述；
9. README、AGENTS、data catalog、registry 和论文更新为 v11 口径。

本报告对应提交将在最终校验后记录。工作区在审计开始前已有上一轮未提交修改；提交必须只声明本轮可审计修复，不能把全部 dirty diff 归因于本审计。

## 6. 尚未验证的风险

- graph-cache-v5 尚未在全部 4998 GMTNet 结构上重建并与外部邻居 oracle 做分层统计；
- 新 4944/1595 split 尚未产生 structural pretrain 或 validation/development 结果；
- A0/A1/A1.5 的参数容量不同，模型类结论需结合参数量、速度、峰值显存共同解释；
- microbatch 改变时存在有限精度 AdamW 敏感性，不能跨不同 microbatch schedule 做无条件比较；
- frozen val/test 自身共享 `HNaO`，除非建立新的 benchmark 版本，否则不能声称完整三方 formula-OOD；
- 当前测试是强针对性 oracle 集，不是对全部空间群/PBC 输入的形式验证；
- historical replay 脚本可被用户显式调用，registry/论文边界仍依赖操作者遵守 legacy 标记。

## 7. 下一步最小必要实验

1. 不读 test20，在新 canonical split 上运行 1-epoch/少更新 smoke：只验证 graph-cache-v5、split/provenance、保存与恢复，不报告性能。
2. 用 train1595/fold-train-only IDs 重新生成 structural checkpoint；确认旧 checkpoint 被 provenance 拒绝。
3. 在 `electrostatic_development_folds_v2` 上运行 matched A0/A1/A1.5；固定同一 pretrain、logical/microbatch schedule、shuffle seed、updates、AdamW 和 development selection，同时报告参数量/时间/显存。
4. 只在 development 改善后运行 validation10；仍不读 test20。历史 train1603/4961 数字不得与新 split 直接合并或作学习曲线。
5. 单独做 microbatch-size 1 vs 2/4 的数值敏感性，判断 AdamW 近零梯度放大是否改变 selection；不得在不同 schedule 间把差异解释为架构收益。
