# PiezoJet 工作总结：2026-07-15 09:00 至本文档生成时

## 一句话结论

本时间段的核心工作是把此前审计指出的**数据接口、可追溯性、严格
completion 数值稳定性和外部只读核验**落实到代码与可复现产物中；并未
获得新的正向泛化性能结果。BEC 轴向与 OUTCAR 内部应变符号的约定已被更
严格地固化和交叉检查，但当前 train149/validation10/test20 的三种子结果
仍不能支持 PiezoJet 已具有有效的 held-out 响应预测能力。

本文只陈述已实现和已运行的事实。`outputs/` 下的实验产物与 `.pt` 缓存
是可再生工件；任何 JARVIS strict-completion 队列都没有修改 frozen 20-ID
测试面板，也没有混入 Materials Project 标签。

## 0. 本轮范围与证据位置

| 范围 | 主要实现/证据 |
| --- | --- |
| DFPT 原始数据可追溯性 | `src/piezojet/jarvis_dfpt.py`；`data/processed/jarvis_dfpt_v5_provenance/manifest.json` |
| 严格内部应变 completion 数值审计 | `src/piezojet/strain_completion.py`；`data/processed/jarvis_strain_completion_v8_provenance/manifest.json` |
| 独立解析器核验 | `src/piezojet/crossvalidate_vasp_parsers.py`；`outputs/pymatgen_phonopy_parser_crosscheck_v2/report.json` |
| PBC / force-constant 外部只读审计 | `tests/test_pbc_graph_ase_oracle.py`；`src/piezojet/symfc_force_constant_audit.py`；`outputs/symfc_force_constant_audit_v1/report.json` |
| 模型候选 | `src/piezojet/model.py` 的 `AnisotropicLinearResponseBackground` |
| 论文 | `E:/PAPER/piezojet_equivariant_response_jets/piezojet.tex`；编译结果 `output/pdf/piezojet.pdf` |

已使用的运行时始终为：

```powershell
$env:PYTHONPATH = 'E:\CODE\PiezoJet\src'
& 'D:\Anaconda\envs\EGNN\python.exe' ...
```

## 1. 数据接口、缓存与来源可追溯性

### 1.1 schema-4 provenance cache

`src/piezojet/jarvis_dfpt.py` 中的 `DFPT_CACHE_SCHEMA` 已从 3 升为 4。
schema 2/3 仍可读取，但仅作为历史格式；新写入的 schema-4 记录不允许
缺少 provenance。

对每一条新缓存记录，代码会保存：

- 原始压缩 archive、`vasprun.xml`、`OUTCAR` 的 SHA256；
- `jarvis-tools` 版本、可选 `JARVIS_TOOLS_COMMIT`、Vasprun parser 源码
  SHA256 与 parser schema；
- `phonon_data(fc_mass=False)` 原始 dynamical-matrix 路径和
  `phonon_data(fc_mass=True)` 质量换算路径的显式语义；
- source/internal BEC、力常数、质量、介电张量等数组的 checksum；
- 单位、BEC 轴向变换、Lambda 符号和坐标约定。

`src/piezojet/migrate_dfpt_conventions.py` 同步更新：历史 schema-2 迁移到
schema-4 时会标为 `legacy_migration_without_raw_archive`，不会伪造下载文件
的 hash。

实际已生成：

```text
data/processed/jarvis_dfpt_v5_provenance/
  manifest.json: schema=4, requested=179, cached=179, failed=[]
```

这里的 179 是当前严格 completion 的记录数，不把它误称为重新审计了全部
610 archive。610 archive 的 convention cache 仍保留在
`data/processed/jarvis_dfpt_v4_conventions/`。

### 1.2 BEC 与内部应变的物理约定被固定到入口处

这不是通过全局补丁“调好”闭合，而是把 VASP/JARVIS source 量和内部算子
所需的行列方向清楚区分：

\[
Z^{\mathrm{source}}_{ij}=\frac{\partial P_i}{\partial u_j},\qquad
Z^{\mathrm{internal}}_{ji}=Z^{\mathrm{source}}_{ij}.
\]

因此 PiezoJet 的 atom-coordinate response 算子使用坐标为行、场/极化为列
的 BEC。OUTCAR 打印的内部应变块按

\[
\Lambda=\frac{\partial F}{\partial\eta}
\]

保存，不再额外施加全局 Lambda 符号翻转。已有 179 条严格样本的
same-source ionic closure 中，转置 BEC 的中位相对残差为 `6.3627e-4`；不转
置为 `5.74e-3`；两种全局 Lambda 取负方案的残差约为 2。因此，代码拒绝了
“统一 Voigt / engineering shear / improper rotation 修补”这类没有受控复现
器的做法。

### 1.3 v8 strict completion：标签集合未扩张，审计信息扩张

`data/processed/jarvis_strain_completion_v8_provenance/` 从 schema-4 cache
重新生成。严格门槛没有放宽，仍为 179/179 accepted；它与
`jarvis_strain_completion_v7_bec_transpose/` 有完全相同的 179 个 JID。两版
`internal_strain_full` 的最大逐元素差为 `3.814697e-06`（存储单位），没有
产生新样本或把历史标签与新约定混合训练。

v8 额外持久化条件数、稳定性分层、exact closure、delta 敏感性和显示精度
bootstrap，具体数值见第 3 节。

## 2. 数学实现与约定核对

### 2.1 translation-free optical operator

代码继续在去掉三条整体平移后的 (3N-3) 光学子空间求解，而不是构造固定
latent mode 或物化稠密逆矩阵。令 (Q^\mathsf{T}T=0)、
\(\Phi_o=Q^\mathsf{T}\Phi Q\)，稳定且可逆时的 stationary operator 为

\[
\mathcal O_0(\Phi)=Q\Phi_o^{-1}Q^\mathsf T.
\]

对 soft-positive 或 unstable 光学谱，生产/训练所用的连续有界 signed
resolvent 是

\[
\mathcal D_\delta(\Phi)=Q\operatorname{Re}
\left[(\Phi_o+i\delta I)^{-1}\right]Q^\mathsf T,
\]

其单本征值滤波为 \(\lambda/(\lambda^2+\delta^2)\)，保留 DFPT 负模符号。
stable exact 与 signed diagnostic 的含义分开记录：后者不是静态平衡响应的
声称。

离子路径仍是物理坐标中的

\[
e^{\mathrm{ion}}\propto
\Omega^{-1}Z^{*\mathsf T}\mathcal O(\Phi)\Lambda,
\]

其中比例系数包含单位换算；它不是对固定 latent eigenmode 的监督。

### 2.2 严格 completion 的可识别性与条件数

对于 \(\ell=\operatorname{vec}_V(\Lambda)\in\mathbb R^{18N}\)，实现保持
space-group 不变性和 acoustic nullspace：

\[
[\rho(g)-I]\ell=0,\qquad (T^\mathsf T\otimes I_6)\ell=0.
\]

设 (B) 是两个 nullspace 交集的正交基、(M_{\cal O}) 选择 OUTCAR 已打印
块，则只有

\[
\operatorname{rank}(M_{\cal O}B)=\dim(B)
\]

时才允许唯一最小二乘 completion，解为

\[
\widehat\ell=B(M_{\cal O}B)^+y_{\cal O}.
\]

本轮在 `strain_completion.py` 中新增/落实的数值项：

- $\sigma_{\min}(M_{\cal O}B)$、$\sigma_{\max}(M_{\cal O}B)$、condition
  number 和 $\lVert(M_{\cal O}B)^+\rVert_2$；
- 对 stable 样本，除原有 signed-operator ionic closure 外，**强制** exact
  inverse closure；
- unstable 样本保留 signed diagnostic，不被改写为 stationary response；
- $\delta\in\{10^{-4},10^{-3},10^{-2}\}$ 的 regularized closure
  sensitivity；
- 根据 OUTCAR 最后一位显示精度的均匀区间 bootstrap。默认不虚构噪声，只有
  明确指定 samples 时才运行；
- 可选的 prospective `--max-condition-number` gate。默认是 `null`，没有
  在看到结果后新设阈值来操纵样本数。

### 2.3 engineering-shear golden test

`tests/test_dfpt_conventions.py` 新增对 canonical order
`(xx, yy, zz, yz, xz, xy)` 中全部三个 shear 列 `(yz, xz, xy)` 的受控测试。
每列同时核验：

1. 标量二次能量；
2. autograd force 与有限差分；
3. OUTCAR 约定 `dF/deta = Lambda`；
4. optical exact solve；
5. BEC 位移导致的极化有限差分；
6. 与离子 (Z^{*\mathsf T}\Phi^{-1}\Lambda) 表达的一致性。

这消除了“只测一个剪切分量、另外两列可能在 Voigt 映射上有因子错误”的
盲点；但它是实现层 golden test，不是对真实 JARVIS archive 的 finite-field
独立验证。

## 3. v8 strict-completion 的实际数值结果

本表来自 179 个 accepted v8 records，rounding bootstrap 使用 64 samples、
固定 seed `20260715`。

| 指标 | 结果 | 正确解释 |
| --- | ---: | --- |
| stable exact | 158 | 必须通过 stationary exact closure |
| unstable signed diagnostic | 21 | 只保留 signed-resolvent 诊断，不称静态平衡 |
| soft-positive | 0 | 本队列没有该 stratum |
| `sigma_min(MB)` median | `0.288675` | 已记录的识别条件数信息，不是新增筛选阈值 |
| condition number median / p95 / max | `1.0000 / 2.7110 / 3.0551` | 没有 post-hoc cutoff |
| `||(MB)^+||_2` median / max | `3.4641 / 6.9282` | 量化 completion 放大风险 |
| ionic closure median / p95 / max | `6.3627e-4 / 2.8102e-2 / 3.9859e-2` | 所有值仍在原有 `0.05` gate 内 |
| stable exact closure max | `3.9859e-2` | stable exact 强制门也满足 `0.05` |
| Lambda bootstrap relative p95 median / max | `5.5107e-7 / 9.6562e-6` | 只由 OUTCAR 显示末位区间引入的敏感性 |

结论：数据重建没有通过降低 strict thresholds 来增加样本数；它使 179 条
标签的数值识别条件、稳定性和打印舍入敏感性都成为可审计字段。它也没有
产生新的模型训练/测试结果。

## 4. 外部工具核验与所安装依赖

### 4.1 安装与环境完整性

在指定 `D:\Anaconda\envs\EGNN\python.exe` 环境安装并核验：

| 包 | 版本 | 用途 |
| --- | ---: | --- |
| `phonopy` | 4.3.1 | 独立读取 VASP XML/OUTCAR 的 BEC、epsilon、FC 路径 |
| `py4vasp` | 0.11.3 | 为后续带 `vaspout.h5` archive 的独立读取预留 |
| `symfc` | 1.7.3 | 只读 force-constant symmetry/ASR 投影审计 |

`pip check` 返回 `No broken requirements found`。

### 4.2 Pymatgen / Phonopy parser cross-check

新增：

- `src/piezojet/crossvalidate_vasp_parsers.py`
- `tests/test_crossvalidate_vasp_parsers.py`
- `outputs/pymatgen_phonopy_parser_crosscheck_v2/report.json`

做法是重新下载 10 个 JARVIS archive，以 Pymatgen `Vasprun/Outcar` 及
Phonopy 的 VASP interfaces 重新解析，并与 PiezoJet ingestion 数组比较。10/10
通过预先登记的 tolerance。中位相对误差：

| 路径 | 中位相对误差 |
| --- | ---: |
| Pymatgen raw dynamical matrix | `2.4498e-8` |
| Pymatgen source BEC | `2.1322e-6` |
| Pymatgen total OUTCAR piezo | `2.3860e-8` |
| Phonopy source BEC | `2.3831e-8` |
| Phonopy epsilon | `2.3346e-8` |
| Phonopy OUTCAR force constants | `5.8005e-4` |

最后一项较大但符合 OUTCAR 打印精度路径。该实验证实**不同 parser 对相同
XML/OUTCAR 文本的读取一致**，不能表述为 phonopy finite-displacement 验证、
finite-field 验证，或 Gamma non-analytic/LO--TO boundary semantics 的证明。
`py4vasp` 没有被误报为完成验证：当前 archive 没有 `vaspout.h5`，因而不能
走 py4vasp 的原生数据路径。

### 4.3 ASE PBC graph oracle

新增 `tests/test_pbc_graph_ase_oracle.py`，用 ASE
`primitive_neighbor_list("ijS")` 做外部参考，覆盖：

- 单原子 cubic cell 的 periodic image shell；
- 小型、多原子和非正交三类真实晶胞；
- PiezoJet retained PBC edges 都是 ASE 认可的有效邻接边。

这是图拓扑的 oracle test，不改变模型截断策略，也不声称 ASE 与实现逐边完全
同构以外的物理性质。

### 4.4 symfc read-only audit

新增：

- `src/piezojet/symfc_force_constant_audit.py`
- `tests/test_symfc_force_constant_audit.py`
- `outputs/symfc_force_constant_audit_v1/report.json`

对 10 个 primitive-cell Gamma force-constant block 做**只读** symfc projection。
结果：projection 相对变化的中位数为 `2.3678e-4`；source ASR 残差中位数为
`5.3850e-4`；symfc projection 后约为 `5.6e-16`。`JVASP-10190` 的 change
为 `0.03259`，是应保留并复核的 outlier。

关键限制：primitive-cell Gamma block 不是完整 real-space force-constant model。
所以 projection 不得替换 cache Phi、不得修改标签，也不能据此宣称 source
record 错误。代码与文档均把它限制为 diagnostic。

## 5. 模型结构清理

各向异性 background 曾作为未训练候选实现，但没有 validation evidence。为避免
形成生产回退分支，本轮已将它和 architecture switch 一并删除；生产代码只保留
isotropic background。介电/弹性张量转换移入
`src/piezojet/elastic_dielectric_ops.py`，误导性的 `response_ops.py` 兼容转发层也已
删除。`tests/test_elastic_dielectric_ops.py` 只验证仍在使用的转换、能量等价和 SPD
定义，不再为未选择的架构维护测试负担。

## 6. 实验、训练与性能结果

### 6.1 本时间段新完成的实验/审计

| 项目 | 是否完成 | 结果 |
| --- | --- | --- |
| schema-4 cache 重新生成 | 完成 | 179/179 cached，带原始输入 provenance |
| v8 strict completion | 完成 | 179/179 accepted，门槛不变；数值见第 3 节 |
| Pymatgen/Phonopy parser audit | 完成 | 10/10 通过，见第 4.2 节 |
| ASE PBC oracle | 完成 | 针对性测试通过 |
| symfc read-only audit | 完成 | 10/10 success；保留 `JVASP-10190` outlier |
| 各向异性背景 | 已清理 | 无 validation evidence，已从生产与候选分支删除 |

### 6.2 当前模型性能上下文（不是本时间段新训练的正向结果）

现有已完成的、validation-loss 选 checkpoint 的 train149/val10/test20 三种子
replay 来自 `outputs/bec_transpose_observable_v1/report/replicates.md`：

| 条件 | Total TRS，mean ± seed SD |
| --- | ---: |
| Observable-subspace PiezoJet | `-0.04416 ± 0.09485` |
| Matched Cartesian direct baseline | `-0.06217 ± 0.08437` |
| Paired difference | `+0.01801 ± 0.17910` |

同一 observable condition 的 factorized ionic material-macro cosine 为
`0.08504 ± 0.17931`，ionic MAE skill 相对 zero 为 `0.00753 ± 0.01641`。
这些值不构成 positive ionic/general response learning 的证据；尤其 paired
interval 很宽，不能声称 factorization 胜出。

以下任务没有被伪报为完成：

- 历史 `run_train149_cartesian_pretrain_replay` 执行器已在结果登记后移除；该实验可消除 108-train
  pretraining capacity confound 的 fresh full-train replay，尚未运行；
- `scripts/run_e3nn_direct_control.ps1`：只完成 e3nn structural pretrain
  20/20 update（final loss `4.63581`）和 direct seed 42 的 1--29/100 update
  （validation loss `0.11822 -> 0.10188`）。无 `test.json`，seed 7/1729 未
  启动，因此没有 e3nn 性能结论；
- pair-vs-local-star 的 preregistered validation-only three-seed comparison
  未完成；
- GMTNet、EATGNN、CEITNet、MACE 等 matched split rerun 未完成。

## 7. 论文更新

论文源位于：

```text
E:/PAPER/piezojet_equivariant_response_jets/piezojet.tex
```

本轮补入：

- schema-4 provenance 的存储范围和不可静默替换训练 cache 的限制；
- v8 的 179 identity、(M_{\cal O}B) conditioning、stable exact closure、
  unstable signed-only 语义与 OUTCAR rounding bootstrap；
- Pymatgen/Phonopy 10 archive parser agreement 的所有中位误差，及其不是
  finite-displacement / finite-field / LO--TO 证据的限制；
- ASE 和 symfc 的 read-only scope；
- canonical shear golden test 与 PBC oracle；
- anisotropic background 的 SPD / equivariance 实现状态和“未评估”限制；
- Pymatgen、Phonopy、ASE 的正式文献引用。

使用：

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=output\pdf piezojet.tex
```

最终 PDF 是 18 页。已刷新 BibTeX 中间文件，检查到没有 unresolved citation
和 overfull horizontal box；关键页面已用渲染 PNG 视觉检查。论文历史上已有的
重复 table hyperlink destination warning 仍存在，未作为本轮科学实现的一部分
修改。

## 8. 最终验证状态

在所有本轮新增测试和论文更新之后，运行：

```powershell
$env:PYTHONPATH = 'E:\CODE\PiezoJet\src'
& 'D:\Anaconda\envs\EGNN\python.exe' -m pytest -q
```

结果：

```text
105 passed, 403 warnings in 21.35s
```

warnings 的主要来源是 PyTorch 对 `torch.jit.script` 的 deprecation、TorchScript
empty-type annotation，以及 symfc 的未来构造器参数弃用提示；没有测试失败。

## 9. 仍未解决、不能弱化为“完成”的问题

1. **显式 long-range electrostatics 未实现。** 当前只有 learned reciprocal
   context；尚未验证 JARVIS Gamma FC 的 non-analytic boundary semantics，也未
   实现/验证 $\Phi_{\mathrm{SR}}+\Phi_{\mathrm{LR}}(Z^*,\epsilon_\infty)$。
2. **没有真实 finite-displacement 或 finite-field cross-validation。** Phonopy
   parser audit 只重读同一 VASP 文本；py4vasp 路径受 archive 缺少 `vaspout.h5`
   限制。
3. **模型泛化瓶颈仍在。** data convention 更正确，但三种子 total TRS 仍为负；
   不能以 data audit 代替 predictive result。
4. **外部 baseline 尚不完整。** e3nn control 中断且没有测试；成熟模型的 matched
   split rerun 未完成。
5. **工程性能仍有空间。** forward path 已使用
   `AtomCoordinateResponsePotential.apply_optical_operator` 避免 dense inverse，
   但 atom-count bucketed batched factor solve、reciprocal vectorization 和
   precision/throughput benchmark 尚未完成。

## 10. 下一步的正确优先级

1. 为 v8 provenance cohort 生成新的、显式版本化 split/config，保证 validation/
   test IDs 与 v1 byte-for-byte 相同，然后注册新的 replay；不覆盖历史 v1/v7
   结果。
2. 在独立 output cohort 运行 fresh train149 Cartesian-pretrain replay，消除旧
   108-train initialization capacity confound。
3. 从新目录启动完整三种子 e3nn direct control；任何结果都只在 test 最后一次
   读取后报告。
4. 若要实现 long-range Φ 分解，先取得有足够原始产物的受控 VASP / phonopy
   reproducer，确认 non-analytic/LO--TO 的边界语义，再写入训练标签或 operator。
5. 对 anisotropic background 和 pair-vs-local-star 都先做 preregistered、
   validation-only 多种子对照；没有此证据前不改生产默认配置。
