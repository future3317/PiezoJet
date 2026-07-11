# PiezoJet 后续完善任务书：从 M2 到 M5 的完整实现与实验

> 本任务书基于当前已完成的 PiezoJet MVP。  
> 当前实现并非错误：它已经正确完成周期晶体图、O(3) 等变压电张量预测、Voigt/工程剪切转换、标量双线性响应势和 M2 小样本拟合。  
> 后续目标是**增量完善现有代码**，而不是推翻重写。

---

# 0. 当前状态与实现判断

当前模型执行：

\[
x\longrightarrow \widehat e_\theta(x),
\]

并定义：

\[
\Phi_\theta(x,E,\eta)
=
-E_i\widehat e_{ijk}(x)\eta_{jk}.
\]

因此：

\[
-\frac{\partial^2\Phi_\theta}
{\partial E_i\partial\eta_{jk}}
=
\widehat e_{ijk}(x).
\]

这是一种正确的、结构化的第一阶段实现。它已经保证：

- 压电张量满足 \(e_{ijk}=e_{ikj}\)；
- 晶体旋转后输出按 rank-3 tensor 旋转；
- 势函数在晶体、应变和电场联合旋转下保持不变；
- 混合导数交换；
- 张量可由 `CartesianTensor("ijk=ikj")` 的 18 维 irreps 表示。

需要明确的是：当前混合导数一致性来自双线性势的结构，而不是模型从多个物理响应数据中自行发现。后续 M5 会加入介电和弹性，并进一步实现真正以外场为输入的 scalar-potential 实验分支。

---

# 1. 总工程原则

必须在当前仓库和代码结构上增量修改。

## 1.1 不允许

- 不重新创建一个新项目；
- 不重写已通过测试的周期图和等变网络；
- 不复制 e3nn、PyG 或 pymatgen 的功能；
- 不引入 Lightning、Hydra、wandb 或复杂 registry；
- 不增加多层自动 fallback；
- 不在下载失败后偷偷改数据源；
- 不用随机数据代替真实实验；
- 不为了跑通而删除当前严格测试；
- 不把 M2 的训练同 cohort 曲线称为泛化结果；
- 不自动搜索 checkpoint；
- 不保存每个 epoch 的 checkpoint。

## 1.2 应当

- 保持当前 CLI 和配置兼容；
- 使用小而清晰的函数；
- 所有 split、数据 hash、单位和 convention 可追踪；
- 关键实验至少运行 3 个 seed；
- 新增功能先写测试再跑完整训练；
- 阻塞时停止并生成明确报告；
- raw prediction 和 symmetry-projected prediction 分开报告；
- 所有指标同时保存 JSON/CSV 和可读 Markdown。

---

# 2. 第一阶段：代码与实验状态审计

在继续训练前，新增一个可重复的审计命令：

```bash
python -m piezojet.audit \
  --config config.yaml \
  --output outputs/audit
```

输出：

```text
outputs/audit/
├── environment.json
├── repository.json
├── data_manifest.json
├── split_manifest.json
├── tensor_convention.json
└── audit.md
```

## 2.1 repository.json

记录：

- 当前 Git commit；
- 是否有未提交改动；
- Python 版本；
- PyTorch、PyG、e3nn、pymatgen、spglib 版本；
- CUDA 版本；
- GPU 名称；
- hostname 可省略，避免不必要隐私。

## 2.2 data_manifest.json

记录：

- 原始数据文件路径；
- 文件 SHA256；
- 原始记录数；
- 有效记录数；
- 过滤规则；
- 过滤数量；
- tensor 字段名；
- tensor 原始 shape；
- tensor 单位；
- 原始 Voigt 顺序；
- 项目内部 Voigt 顺序；
- engineering shear 约定；
- 中心对称样本数；
- 非中心对称样本数；
- 重复 material id 数；
- NaN/Inf 数。

## 2.3 split_manifest.json

记录：

- seed；
- split 方法；
- train/val/test 数量；
- material id 列表文件；
- formula overlap；
- chemical-system overlap；
- structure hash overlap；
- split 文件 SHA256。

如果现有 split 已经存在，只读取，不重新生成。

---

# 3. M2.1：更严格的 memorization test

当前 M2 已通过。新增 M2.1 的目的不是追求论文指标，而是进一步排除隐性实现限制。

## 3.1 实验设置

- 固定当前 32 个真实样本；
- train 和 evaluation 使用相同 cohort；
- 单 batch 或最多两个 batch；
- `weight_decay = 0`；
- 所有 dropout 关闭；
- 不使用 early stopping；
- 训练 300 epoch；
- seed = 42；
- full tensor loss；
- 保存 best 和 last；
- 每 10 epoch 输出一次详细统计。

## 3.2 记录

除了 normalized loss，还要记录：

- unnormalized tensor MSE；
- Frobenius error；
- 每个样本的 error；
- 每个 irrep block 的 error；
- gradient norm；
- predicted tensor norm；
- target tensor norm；
- isolated-node 数；
- batch 中最大原子数；
- 是否有梯度为 NaN/Inf。

## 3.3 诊断目标

理想目标：

\[
\mathcal L_{\mathrm{norm}}<10^{-3}.
\]

这不是硬性通过阈值。如果训练稳定但停在更高值，不要立即改架构，应输出：

```text
outputs/m2_1/plateau_diagnosis.md
```

至少分析：

1. 哪些样本占主要误差；
2. 哪些 irrep block 难拟合；
3. 高张量范数样本是否主导；
4. cutoff 图是否截断重要邻居；
5. readout 使用 mean 或 sum 的影响；
6. odd/even parity channel 是否齐全；
7. normalization 是否正确反归一化；
8. tensor conversion 两端是否完全一致。

只有明确证据表明架构表达受限时才修改模型。

## 3.4 命名规范

训练和同 cohort 评测不得称为 validation generalization。

建议日志字段：

```text
optimization_loss
memorization_loss
```

---

# 4. M3：固定随机 split 上的完整泛化实验

使用现有固定 split：

```text
train: 3998
val:    500
test:   500
```

如数量与当前 manifest 不一致，以 manifest 为准并停止检查原因，不静默重分。

## 4.1 训练 seeds

运行：

```text
42
43
44
```

每个 seed：

- 相同 split；
- 相同 normalization；
- 相同训练预算；
- early stopping 只看 validation；
- 测试集只在最终 best checkpoint 上评估一次；
- 保存 best 和 last；
- 不针对 test 调参。

## 4.2 主模型

当前完整 PiezoJet：

- PBC radius graph；
- cutoff 5 Å；
- \(l_{\max}=3\)；
- 3 层 tensor-product message passing；
- CartesianTensor piezo head；
- full tensor loss。

除非 M2.1 明确发现问题，否则 M3 不修改架构。

---

# 5. 必须实现的评测指标

所有指标同时报告：

1. 全测试集；
2. 非中心对称测试集；
3. 中心对称测试集；
4. 不同张量强度分桶；
5. 不同点群或晶系；
6. 高响应尾部。

设单样本张量误差：

\[
\Delta e_n=\widehat e_n-e_n.
\]

## 5.1 Component MAE

\[
\mathrm{ComponentMAE}
=
\frac1{18N}
\sum_n
\|\Delta e_n\|_1.
\]

单位必须写清楚。

## 5.2 Frobenius RMSE

\[
\mathrm{FrobRMSE}
=
\sqrt{
\frac1N
\sum_n
\frac{\|\Delta e_n\|_F^2}{18}
}.
\]

## 5.3 Sample-wise Frobenius MAE

\[
\mathrm{FrobMAE}
=
\frac1N
\sum_n
\frac{\|\Delta e_n\|_F}{\sqrt{18}}.
\]

## 5.4 Stabilized relative error

不要直接除以接近零的标签。

令：

\[
\tau=0.05\,s_e
\]

或使用训练集 tensor RMS 的 5%，保存该值。

\[
\mathrm{RelErr}_\tau
=
\frac1N
\sum_n
\frac{
\|\Delta e_n\|_F
}{
\max(\|e_n\|_F,\tau)
}.
\]

## 5.5 最大分量误差

\[
\mathrm{MaxComponentMAE}
=
\frac1N
\sum_n
\left|
\max_{iJ}|e_{n,iJ}|
-
\max_{iJ}|\widehat e_{n,iJ}|
\right|.
\]

## 5.6 中心对称假阳性

\[
\mathrm{CentroFP}
=
\frac1{|\mathcal D_c|}
\sum_{n\in\mathcal D_c}
\|\widehat e_n\|_F.
\]

同时报告：

- median；
- 90th percentile；
- max；
- 超过 \(10^{-4},10^{-3},10^{-2}\) 的比例。

## 5.7 非中心对称主指标

\[
\mathrm{FrobMAE}_{nc}
=
\frac1{|\mathcal D_{nc}|}
\sum_{n\in\mathcal D_{nc}}
\frac{\|\Delta e_n\|_F}{\sqrt{18}}.
\]

该指标必须在 summary 顶部单独出现。

## 5.8 强度分桶

按训练集目标 \(\|e\|_F\) 的 quantile 固定阈值：

- zero / centrosymmetric；
- nonzero 0–25%；
- 25–50%；
- 50–75%；
- 75–90%；
- 90–99%；
- top 1%。

不要用测试集 quantile 定义阈值。

## 5.9 高响应排序

对测试集按 target \(\|e\|_F\) 排序，报告：

- top-10% recall；
- top-5% recall；
- Spearman correlation；
- Kendall tau；
- predicted top-k 中真实 top-k 的比例。

---

# 6. 对称性后处理：Reynolds projector

当前网络具有 O(3) 等变性，但并不自动保证数值输出严格落在每个结构点群允许的张量子空间。

实现可选的点群投影：

\[
\Pi_G(e)
=
\frac1{|G|}
\sum_{g\in G}
\rho_3(R_g)e.
\]

其中：

\[
[\rho_3(R)e]_{ijk}
=
R_{ia}R_{jb}R_{kc}e_{abc}.
\]

## 6.1 实现

新增尽量小的函数，不创建新子系统：

```python
get_cartesian_point_group_operations(structure)
project_piezo_to_point_group(tensor, rotations)
point_group_residual(tensor, rotations)
```

优先使用：

```python
pymatgen.symmetry.analyzer.SpacegroupAnalyzer
```

或现有 spglib 结果。

必须明确：

- rotations 是 Cartesian 还是 fractional；
- 是否包含 improper rotations；
- rank-3 polar tensor 在反演下如何变换；
- tolerance；
- 所用标准化结构。

## 6.2 指标

投影前：

\[
r_G(e)
=
\frac{
\|e-\Pi_G(e)\|_F
}{
\|e\|_F+\epsilon
}.
\]

分别报告：

- label residual；
- raw prediction residual；
- projected prediction residual；
- raw metrics；
- projected metrics。

不要只报告 projected 结果。

## 6.3 训练消融

完成 evaluation-only projector 后，可增加：

```yaml
symmetry_projection:
  mode: none | loss | output
```

- `none`：原始模型；
- `loss`：loss 前投影 prediction；
- `output`：只在 evaluation 投影。

默认保持 `none`，以便公平比较。

## 6.4 中心对称 sanity check

含反演的点群投影后，rank-3 polar piezo tensor 应为零。

写单元测试验证：

\[
\|\Pi_G(e)\|_F<10^{-6}
\]

对随机输入 tensor 和含 inversion 的操作集合成立。

---

# 7. M3 基线

所有基线使用完全相同的 split 和数据转换。

## 7.1 Zero baseline

\[
\widehat e=0.
\]

必须报告全体和非中心对称子集。

## 7.2 Mean baseline

训练集 tensor 的均值。  
由于对称性，Cartesian component-wise mean 可能没有物理意义，因此只作为统计基线，并明确标注。

## 7.3 Composition-only baseline

输入：

- 元素种类；
- 元素计数或原子分数；
- 不使用坐标、晶格和邻接图。

建议实现为：

- atomic number embedding；
- composition-weighted mean；
- 两层 MLP；
- 输出相同 18 维 tensor representation。

该 baseline 不需要等变，因为它没有方向信息。预期它只能学到接近零或统计先验；其价值是验证几何信息是否必要。

## 7.4 Direct 18-scalar baseline

尽量复用当前 periodic encoder 的 scalar graph embedding，直接输出 18 个数，不使用 `CartesianTensor` 约束。

为了公平：

- 参数量尽量接近；
- 使用相同训练预算；
- 同样转换回 \(3\times6\) 评测；
- 明确它不具备严格旋转等变性。

为该 baseline 做旋转测试，预期 residual 不为零。

## 7.5 GMTNet 官方 baseline

尝试使用 GMTNet 官方代码和相同发布数据。

要求：

1. 不改写 GMTNet；
2. 记录官方 commit；
3. 按其 README 安装；
4. 尽量使用当前相同 split；
5. 如果官方代码只支持自己的 split，额外报告但不要混为公平比较；
6. 安装或接口阻塞时停止该 baseline，并输出：

```text
outputs/baselines/gmtnet/BLOCKED.md
```

不要为了“必须有结果”静默替换实现。

---

# 8. 数据泄漏和重复审计

实现一个独立脚本或 audit 子命令，输出：

## 8.1 material id

- 重复 ID；
- 同一 ID 多条不同 tensor；
- 同一 ID 多条相同 structure。

## 8.2 reduced formula overlap

统计：

- train–val overlap；
- train–test overlap；
- val–test overlap。

## 8.3 chemical system overlap

例如：

```text
Ba-O-Ti
```

元素排序后作为 group key。

## 8.4 structure hash

构建一个轻量、确定性的结构 hash：

- 标准化元素序列；
- 晶格 metric tensor 四舍五入；
- 分数坐标排序；
- 指定 tolerance。

它只用于快速重复审计，不代替 StructureMatcher。

## 8.5 StructureMatcher 小规模检查

对于 hash 相同或高度相似的候选，再使用：

```python
pymatgen.analysis.structure_matcher.StructureMatcher
```

确认跨 split 近重复。

不要对全部 5,000 样本做无差别 \(O(N^2)\) 比较。

---

# 9. OOD splits

所有 OOD split 都必须持久化为 JSON，不能每次运行重新生成。

## 9.1 Formula-OOD

同一 reduced formula 不跨 split。

要求：

- 以 group 为单位分配；
- 尽量接近 80/10/10；
- 保存每组样本数；
- 报告中心对称比例；
- 报告 tensor norm 分布。

## 9.2 Chemical-system OOD

同一 chemical system 不跨 split。

例如：

```text
Ba-Ti-O
```

这是比 formula 更严格的组成外推。

## 9.3 Prototype-OOD

在 formula 和 chemical-system split 完成后再做。

建议两阶段：

1. 使用匿名化 composition + space group + Wyckoff multiplicity 构造粗 prototype key；
2. 对同 key 内使用 StructureMatcher 聚类。

同一 prototype cluster 不跨 split。

若聚类运行超过合理时间或内存，不写复杂分布式实现。记录性能并停止该项优化，先交付粗 prototype split。

## 9.4 OOD 验收

每个 split 生成：

```text
split.json
statistics.json
overlap_check.json
README.md
```

其中 overlap 必须为零，否则 split 失败。

---

# 10. M4：Full、Sketch 与 Hybrid 的完整比较

重要：当前模型显式输出 18 维压电张量，full loss 本身很便宜。  
因此 sketch 是否更快、更省显存必须通过实验确认，不允许预设结论。

## 10.1 实现三种路径

### A. Full

\[
\mathcal L_{\mathrm{full}}
=
\frac1B
\sum_n
\|\widehat e_n-e_n\|_F^2.
\]

### B. Direct sketch

随机：

\[
a\in\mathbb R^3,\qquad
b\in\mathbb R^6.
\]

\[
s_{\mathrm{direct}}
=
a^\mathsf T\widehat e\,b.
\]

直接从预测 tensor 计算，不经过 JVP。

### C. JVP sketch

通过势函数：

\[
s_{\mathrm{jvp}}
=
-
\left.
\frac{\partial^2}
{\partial t\,\partial r}
\Phi(x,ta,rb)
\right|_{t=r=0}.
\]

使用嵌套 `torch.func.jvp`。

## 10.2 等价性测试

必须验证：

\[
|s_{\mathrm{direct}}-s_{\mathrm{jvp}}|<10^{-5}.
\]

还要比较参数梯度：

\[
\frac{
\|g_{\mathrm{direct}}-g_{\mathrm{jvp}}\|_2
}{
\|g_{\mathrm{direct}}\|_2+\epsilon
}
<10^{-4}.
\]

若某些 op 不支持 JVP：

- 先做最小复现；
- 尽量替换为 PyTorch 原生 op；
- 不写复杂 custom autograd；
- 仍阻塞则报告。

## 10.3 Sketch 分布

比较：

### Gaussian

\[
a_i,b_j\sim\mathcal N(0,1).
\]

### Rademacher

\[
a_i,b_j\in\{-1,+1\}
\]

等概率。

两者满足：

\[
\mathbb E[aa^\mathsf T]=I,\qquad
\mathbb E[bb^\mathsf T]=I.
\]

## 10.4 每样本 sketch 数量

运行：

```text
k = 1, 2, 4, 8
```

损失为多个 sketch 平均。

## 10.5 Hybrid

\[
\mathcal L_{\mathrm{hybrid}}
=
\mathcal L_{\mathrm{sketch}}
+
\lambda_f\mathcal L_{\mathrm{full}}.
\]

比较：

```text
lambda_f = 0.01, 0.1
```

不要做大规模超参搜索。

## 10.6 Gradient fidelity

固定同一个 batch 和模型参数，计算：

\[
\cos(g_s,g_f)
=
\frac{
g_s^\mathsf T g_f
}{
\|g_s\|_2\|g_f\|_2+\epsilon
}.
\]

每种 sketch 设置统计 100 次：

- mean；
- std；
- 5th percentile；
- 95th percentile。

## 10.7 资源测量

每种方法测量：

- forward time；
- backward time；
- total step time；
- samples/s；
- `torch.cuda.max_memory_allocated()`；
- `torch.cuda.max_memory_reserved()`；
- CPU RAM 可选；
- 达到固定 validation loss 的 wall-clock；
- 最终 test metrics。

测量前：

```python
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
```

测量后再 synchronize。

预热至少 10 step，统计至少 100 step。

## 10.8 公平实验

所有方法：

- 相同初始化；
- 相同 batch 顺序；
- 相同 split；
- 相同 optimizer；
- 相同 epoch；
- 相同 seeds 42/43/44。

## 10.9 结论规则

如果 JVP sketch：

- 更慢；
- 更占显存；
- 精度下降；
- 方差明显更高；

则如实保留结果，不为维护 idea 强行优化。

同时保留 direct sketch，以区分：

- 随机投影本身的统计代价；
- 高阶自动微分实现的计算代价。

---

# 11. M5 数据审计：介电、弹性和压电

在增加模型前，先执行：

```bash
python -m piezojet.audit_responses \
  --data-root data/raw/gmtnet \
  --output outputs/response_audit
```

输出：

```text
response_fields.json
response_intersections.json
response_units.md
response_examples.pt
```

## 11.1 必须确认

对每类响应：

- 文件名；
- 字段名；
- shape；
- 单位；
- Voigt 顺序；
- symmetry；
- material id；
- 缺失比例；
- NaN/Inf；
- 数值范围；
- outlier 过滤规则；
- 是否和同一结构/同一 DFT 计算对应。

不允许根据字段名猜单位。

## 11.2 交集

统计：

\[
N_e,\quad N_C,\quad N_\chi,
\]

\[
N_{e\cap C},\quad
N_{e\cap\chi},\quad
N_{C\cap\chi},\quad
N_{e\cap C\cap\chi}.
\]

还要统计三个交集中的：

- 中心对称比例；
- 元素覆盖；
- 晶系覆盖；
- tensor norm 分布。

## 11.3 数据阻塞

若 elastic 或 dielectric tensor convention 无法从数据和官方 loader 唯一确定：

- 不实现 head；
- 创建 `RESPONSE_DATA_BLOCKED.md`；
- 指明需要用户确认的字段或文件。

---

# 12. M5A：共享 encoder + 系数化统一响应势

这是 M5 的稳定实现路径。

学习：

- 压电 \(e\)；
- 弹性 \(C\)；
- 介电 susceptibility 或 dielectric tensor \(\chi\)。

标量势：

\[
\Phi_\theta
=
\frac12\eta^\mathsf T
\widehat C_\theta(x)\eta
-
E^\mathsf T
\widehat e_\theta(x)\eta
-
\frac12E^\mathsf T
\widehat\chi_\theta(x)E.
\]

## 12.1 Tensor symmetries

### Piezo

\[
e_{ijk}=e_{ikj}.
\]

继续使用：

```python
CartesianTensor("ijk=ikj")
```

### Dielectric

\[
\chi_{ij}=\chi_{ji}.
\]

使用 e3nn `CartesianTensor`，但 symmetry 字符串必须从官方 API 测试确认，不未经验证硬编码。

### Elastic

\[
C_{ijkl}
=
C_{jikl}
=
C_{ijlk}
=
C_{klij}.
\]

同样使用 `CartesianTensor` 表达，不手写 21 维不可约变换。  
先写 round-trip 和 rotation tests，再接模型。

## 12.2 Heads

保持一个共享 periodic encoder，三个小 head：

```text
piezo_head
dielectric_head
elastic_head
```

不要复制三个 encoder。

## 12.3 Missing-label mask

单样本：

\[
\mathcal L_n
=
m_e\lambda_e\mathcal L_e
+
m_C\lambda_C\mathcal L_C
+
m_\chi\lambda_\chi\mathcal L_\chi.
\]

对 batch 中每个任务除以有效标签数，不除以总 batch size：

\[
\mathcal L_e
=
\frac{
\sum_n m_{e,n}\ell_{e,n}
}{
\sum_n m_{e,n}+\epsilon
}.
\]

如果某 batch 某任务没有标签，该项为零并跳过，不产生 NaN。

## 12.4 Task normalization

每种 tensor 使用独立的、保持等变性的 scalar RMS 或 irrep-block RMS。

不得对 Cartesian 分量分别减不同均值。

## 12.5 Loss weighting

第一版使用固定权重，使归一化后三个 task loss 初始量级接近。

可选加入 uncertainty weighting，但不要默认引入复杂多任务算法。

## 12.6 Baselines

比较：

1. piezo-only；
2. 三个完全独立 encoder；
3. shared encoder + independent tensor heads；
4. shared encoder + unified response potential API。

第 3 和第 4 若 tensor head 相同，数值预测可能一致；第 4 的价值在统一导数接口和后续 field-conditioned 分支。报告时不要制造虚假区别。

---

# 13. M5B：真正的 field-conditioned scalar-potential 分支

这是后续更完整的实现，不应在 M5A 未通过时开始。

目标是不先显式输出全部响应 tensor，而是输入外场和应变：

\[
\Phi_\theta(x,E,\eta)\in\mathbb R,
\]

再通过零场导数得到：

\[
\widehat C_{\mu\nu}
=
\left.
\frac{\partial^2\Phi_\theta}
{\partial\eta_\mu\partial\eta_\nu}
\right|_0,
\]

\[
\widehat\chi_{\alpha\beta}
=
-
\left.
\frac{\partial^2\Phi_\theta}
{\partial E_\alpha\partial E_\beta}
\right|_0,
\]

\[
\widehat e_{\alpha\mu}
=
-
\left.
\frac{\partial^2\Phi_\theta}
{\partial E_\alpha\partial\eta_\mu}
\right|_0.
\]

## 13.1 External irreps

电场是 polar vector：

\[
E\in\mathcal H^1_o.
\]

对称应变：

\[
\eta\in\mathcal H^0_e\oplus\mathcal H^2_e.
\]

使用 e3nn 将：

- \(E\)；
- trace strain；
- traceless strain；

转为 irreps，不把 9 个值当普通 scalars 拼接。

## 13.2 最小注入方式

不要重写 encoder。建议：

1. 先用当前 encoder 得到 graph irreps \(h(x)\)；
2. 构造外场 irreps \(q(E,\eta)\)；
3. 使用 2–3 层小型 equivariant tensor-product response network；
4. 输出 `0e` scalar。

形式：

\[
h_0=h(x),
\]

\[
h_{l+1}
=
\mathrm{Gate}
\left(
W_l
\left[
h_l\otimes q(E,\eta)
\right]
+
U_lh_l
\right),
\]

\[
\Phi_\theta
=
\mathrm{Linear}_{0e}(h_L).
\]

必须保证：

\[
\Phi(Rx,RE,R\eta R^\mathsf T)
=
\Phi(x,E,\eta).
\]

## 13.3 零场二阶导数

因为要在 \(E=\eta=0\) 处得到非零二阶导数：

- 网络必须包含至少二阶外场交互；
- 只做一次线性 field injection 不够；
- 写解析 toy test 验证已知 quadratic potential 能被导数恢复；
- 在真实网络训练前，确认二阶导数不全为零。

## 13.4 训练

先在 32 个样本上：

- 只监督 piezo mixed derivative；
- 检查可过拟合；
- 再加 dielectric；
- 最后加 elastic。

不要一次性联合调试三个二阶导数。

## 13.5 与 M5A 比较

比较：

- 精度；
- 显存；
- step time；
- 导数一致性；
- OOD；
- 低标签性能。

M5B 可能明显更慢，结果应如实记录。

---

# 14. Maxwell 和导数一致性测试

## 14.1 M5A

系数化 quadratic potential 中：

\[
\partial_E\partial_\eta\Phi
=
\partial_\eta\partial_E\Phi
\]

由构造成立。

只作为 sanity test。

## 14.2 M5B

分别使用两条 JVP/VJP 顺序：

\[
J_{E\eta}
=
\frac{\partial}{\partial E}
\left(
\frac{\partial\Phi}{\partial\eta}
\right),
\]

\[
J_{\eta E}
=
\frac{\partial}{\partial\eta}
\left(
\frac{\partial\Phi}{\partial E}
\right).
\]

定义：

\[
\mathrm{MaxwellGap}
=
\frac{
\|J_{E\eta}-J_{\eta E}^\mathsf T\|_F
}{
\|J_{E\eta}\|_F+\epsilon
}.
\]

在 float64 toy test 中 `<1e-7`，真实模型 float32 中 `<1e-4`。

---

# 15. 低标签实验

完成 M5A 后，固定同一个 split，只减少训练集中的 piezo 标签：

```text
10%
25%
50%
100%
```

结构样本仍可通过 dielectric/elastic task 参与共享 encoder 训练。

比较：

1. piezo-only；
2. shared encoder + all response labels；
3. three independent encoders；
4. M5B field-conditioned potential，可后做。

每个比例 3 seeds。

必须确保：

- 标签子集在所有方法中完全相同；
- validation/test piezo 标签完整；
- normalization 只使用可见训练标签；
- 不让隐藏 piezo 标签进入 early stopping。

---

# 16. 训练稳定性与数值检查

所有新实验增加：

- gradient clipping 可配置，默认关闭；
- AMP 可配置，默认先关闭；
- float64 仅用于 tensor/JVP 单测；
- 每 epoch 检查 NaN/Inf；
- tensor norm histogram；
- loss per task；
- effective samples per task；
- batch 中空任务比例；
- optimizer state 是否有限。

发生 NaN 时：

1. 保存最后一个正常 batch 的 material ids；
2. 保存输入 tensor statistics；
3. 保存 model/optimizer state；
4. 停止；
5. 不自动降低 learning rate 无限重试。

---

# 17. 测试清单

保留现有全部测试，并新增。

## 17.1 Symmetry projector

- identity projector 不改变 tensor；
- centrosymmetric projector 输出零；
- projector idempotent：

\[
\Pi_G(\Pi_G(e))=\Pi_G(e).
\]

## 17.2 Direct/JVP sketch

- scalar projection 一致；
- parameter gradient 一致；
- Gaussian identity；
- Rademacher identity。

## 17.3 Split tests

- formula-OOD overlap 为零；
- chemical-system overlap 为零；
- prototype group 不跨 split；
- split 可序列化并稳定读取。

## 17.4 Multiresponse round trip

- dielectric Cartesian ↔ irreps；
- elastic Cartesian ↔ irreps；
- rotation equivariance；
- minor/major symmetries。

## 17.5 Missing-label batch

- 任一 task 全缺失时 loss 有限；
- 三个 task 都有标签时梯度均非零；
- mask 不泄漏隐藏标签。

## 17.6 Field-conditioned scalar

- 联合旋转不变；
- 二阶导数非零；
- mixed derivative order 一致；
- toy quadratic potential 恢复已知 \(C,e,\chi\)。

---

# 18. 输出目录

保持简单：

```text
outputs/
├── audit/
├── m2_1/
├── m3/
│   ├── seed_42/
│   ├── seed_43/
│   ├── seed_44/
│   └── aggregate/
├── baselines/
├── ood/
│   ├── formula/
│   ├── chemical_system/
│   └── prototype/
├── m4/
│   ├── full/
│   ├── direct_sketch/
│   ├── jvp_sketch/
│   └── aggregate/
└── m5/
    ├── data_audit/
    ├── coefficient_potential/
    ├── field_conditioned/
    └── low_label/
```

每个实验目录只保留：

```text
best.pt
last.pt
config.resolved.yaml
metrics.csv
summary.json
report.md
```

资源 benchmark 额外保存：

```text
resource_metrics.csv
```

---

# 19. 实施顺序与停止点

严格按顺序。

## M2.1

- 强化 memorization；
- 诊断 plateau。

通过后进入 M3。

## M3

- random split；
- 3 seeds；
- 完整分层指标；
- baselines；
- raw/projected 对比。

如果主模型在非中心对称子集上不优于简单 baseline，先排查，不进入复杂 M5。

## OOD

- formula；
- chemical system；
- prototype。

formula 和 chemical system 必须完成；prototype 可因性能阻塞单独汇报。

## M4

- full；
- direct sketch；
- JVP sketch；
- Gaussian/Rademacher；
- k=1/2/4/8；
- hybrid；
- 资源与梯度比较。

不预设 sketch 胜出。

## M5 audit

- 三类响应数据；
- 单位；
- tensor convention；
- 交集。

数据不明确则停止 M5。

## M5A

- shared encoder；
- three coefficient heads；
- masked heterogeneous training；
- low-label study。

## M5B

- field-conditioned scalar；
- derivative supervision；
- 先 piezo，再 dielectric，再 elastic。

---

# 20. 每个里程碑的交付报告

每个阶段生成一个 `report.md`，固定结构：

```markdown
# Milestone

## Git commit

## Data manifest

## Configuration

## What was implemented

## Tests

## Training behavior

## Metrics

## Resource usage

## Failed or blocked items

## Interpretation boundary

## Exact next step
```

`Interpretation boundary` 必须明确说明该阶段能证明什么、不能证明什么。

---

# 21. 阻塞报告

如果出现依赖、下载、数据 convention 或官方 baseline 问题，创建：

```markdown
# PiezoJet implementation blocked

## Milestone

## Command

## Git commit

## Data file and SHA256

## Exact error

## What was already verified

## No fallback taken

## Needed from the user

## Safe resume command
```

阻塞后不要继续使用替代数据或伪造结果。

---

# 22. 最终验收条件

本轮完善完成应满足：

- 当前 M2 功能不回退；
- M2.1 有严格 memorization 诊断；
- M3 有真实 validation/test 和 3 seeds；
- 指标区分中心/非中心、高响应尾部和 tensor norm 分桶；
- 有 zero、composition-only、direct-scalar 和 GMTNet 尝试；
- 有点群 Reynolds projection 和 raw/projected 对照；
- formula-OOD 和 chemical-system OOD 完成；
- prototype leakage 至少完成粗分组审计；
- full/direct-sketch/JVP-sketch 被严格区分；
- Gaussian、Rademacher、k 和 hybrid 比较完成；
- 有 wall-clock、显存、throughput 和 gradient fidelity；
- 多响应数据交集与单位审计完成；
- M5A masked multiresponse 可训练；
- 低 piezo 标签实验完成；
- M5B 至少完成 piezo derivative 的 32 样本 overfit；
- 所有新增 tensor symmetry、JVP、split 和 mask 测试通过；
- 无静默 fallback；
- 项目结构仍保持简洁。

---

# 23. 给 Codex 的最后执行要求

先阅读现有代码、当前测试和最新 M2 报告，再提交实施计划。

第一条回复只需给出：

1. 现有仓库结构摘要；
2. 将修改的文件；
3. 将新增的文件；
4. M2.1 到 M5 的执行顺序；
5. 预计可能阻塞的数据或依赖；
6. 不会进行的重构。

得到确认后再编码。

实现过程中每完成一个里程碑就提交一个独立 commit，不要把所有改动压在一个巨大 commit 中。  
commit message 示例：

```text
audit: record reproducible data and split manifests
experiment: add strict M2.1 memorization diagnostics
eval: add stratified piezo metrics and symmetry projection
baseline: add zero composition and unconstrained tensor models
data: add formula and chemical-system OOD splits
experiment: benchmark full direct-sketch and jvp-sketch
data: audit dielectric elastic and piezo intersections
model: add masked multiresponse coefficient potential
model: add field-conditioned scalar response prototype
```
