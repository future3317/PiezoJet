# PiezoJet 下一阶段任务书：先完成短周期完善，再启动完整 M3

> 当前基线提交：`1a620ca`  
> 本阶段不急于运行完整的 3-seed M3。先把低成本、能暴露实现问题的任务做完，并冻结模型、数据约定和评测流程，避免之后因代码变化重复消耗多小时 GPU。

---

## 1. 先明确 M2.1、M3、M4 的关系

### M2.1：同 cohort 记忆测试

当前 M2.1 使用 32 个真实样本，训练集和评估 cohort 相同，训练 300 epoch。

它回答：

> 模型、数据管线、张量转换、反向传播和优化器是否能把一小批真实数据拟合下来？

当前最佳 normalized loss：

\[
3.4033\times 10^{-4}.
\]

这说明表达和优化链路基本没有问题，但不说明泛化。

---

### M4 的“100 step”是什么

当前 M4 使用：

- 同一个 32 样本真实 cohort；
- M2.1 checkpoint；
- 10 step warmup；
- 100 个被计时 step；
- full、direct sketch、JVP sketch 三条路径。

它是 **micro-benchmark / resource benchmark**，不是完整训练。

它回答：

> 在相同 batch 和模型状态下，一次 forward + backward step 的速度、吞吐量和峰值显存是多少？

当前结果：

| 路径 | 时间 |
|---|---:|
| full | 5.650 s/step |
| direct sketch | 5.698 s/step |
| JVP sketch | 5.753 s/step |

显存基本一致。

这 100 step：

- 不代表 100 epoch；
- 不代表模型已经重新训练完成；
- 不使用独立 test 集；
- 不产生泛化结论；
- 不替代 M3；
- 主要用于判断 sketch 是否值得继续投入。

当前初步结论是：在显式输出 18 维压电张量的任务上，full loss 最简单且略快，JVP sketch 暂未显示资源优势。

---

### M3 是什么

M3 使用：

```text
train = 3998
val   = 500
test  = 500
```

每个 seed 都需要完整训练，并在 best validation checkpoint 上进行一次 test evaluation。

它回答：

> 模型能否从训练材料泛化到没有参与训练的材料？

完整 M3 的 3 seeds：

```text
42, 43, 44
```

主要是为了估计训练随机性的均值和方差。

---

## 2. 当前决策：3 seeds 不着急

现在不建议立刻投入完整 3-seed 长训练，原因：

1. M5A 还会增加介电、弹性和多任务共享 encoder；
2. 数据单位、tensor converter、mask 和新 head 仍会改变代码；
3. 现在跑完 3 seeds，后续若修复数据或模型细节，可能需要全部重跑；
4. 当前 M2.1 已经证明模型能学习；
5. 当前 M4 已经证明 full/direct/JVP 三条路径可运行；
6. 先完成单元测试、M5A 小样本拟合和 M3 dry-run，更节省时间。

完整 M3 应在以下内容冻结后再跑：

- piezo tensor convention；
- elastic convention；
- dielectric target；
- model architecture；
- normalization；
- evaluation metrics；
- symmetry projection；
- checkpoint resume；
- test-only evaluation；
- aggregate script。

---

# 3. 本阶段执行顺序

严格按以下顺序执行。

---

## Phase A：完成 M4 的短周期收尾

### A1. 完整 gradient fidelity matrix

运行已经实现的 gradient fidelity 脚本：

- Gaussian；
- Rademacher；
- \(k=1,2,4,8\)；
- 100 trials；
- direct sketch；
- JVP sketch；
- 与 full gradient 比较。

记录：

\[
\cos(g_{\mathrm{sketch}},g_{\mathrm{full}})
=
\frac{
g_{\mathrm{sketch}}^\mathsf T g_{\mathrm{full}}
}{
\|g_{\mathrm{sketch}}\|_2
\|g_{\mathrm{full}}\|_2+\epsilon
}.
\]

输出：

```text
outputs/m4/gradient_fidelity.json
outputs/m4/gradient_fidelity.csv
outputs/m4/gradient_fidelity.md
```

每个配置报告：

- cosine mean/std；
- 5th/50th/95th percentile；
- gradient norm ratio；
- direct/JVP scalar gap；
- direct/JVP gradient gap。

### A2. 一个短训练对比即可

不要现在跑 3 seeds 的 full/direct/JVP 长训练。

只运行一个固定 seed 的短训练：

```text
seed = 42
steps = 300 或 500
same initialization
same batch order
same optimizer
```

比较：

1. full；
2. direct Rademacher sketch, \(k=4\)；
3. JVP Rademacher sketch, \(k=4\)；
4. hybrid：sketch + 0.1 full。

记录：

- optimization loss；
- validation loss，可用固定小 validation cohort；
- wall-clock；
- peak memory；
- samples/s。

这只是收敛行为检查，不是 M3。

### A3. M4 停止条件

如果结果继续显示：

- full 更快或相同；
- full 显存相同；
- sketch 方差更高；
- sketch 精度没有优势；

则冻结结论：

> 当前 18 维显式 tensor 输出任务默认使用 full loss；sketch 作为研究性实现和后续高维响应算子的扩展接口保留，但不作为默认训练路径。

不要继续投入大规模 sketch sweep。

---

## Phase B：正式解除 M5 数据约定阻塞

将以下约定写入一个唯一的机器可读配置，例如：

```text
configs/response_conventions.yaml
```

不要把单位和 permutation 分散在多个文件。

### B1. Piezo

```yaml
piezo:
  field: piezoelectric_C_m2
  type: total
  unit: C/m^2
  source_voigt_order: [xx, yy, zz, xy, yz, xz]
  internal_voigt_order: [xx, yy, zz, yz, xz, xy]
  engineering_shear: true
```

### B2. Elastic

```yaml
elastic:
  field: elastic_total_kbar
  type: total
  source_unit: kbar
  target_unit: GPa
  scale_to_target: 0.1
  source_voigt_order: [xx, yy, zz, xy, yz, xz]
  internal_voigt_order: [xx, yy, zz, yz, xz, xy]
  engineering_shear: true
```

使用：

\[
C_{\mathrm{GPa}}
=
0.1C_{\mathrm{kbar}}.
\]

工程剪切：

\[
\eta_V=
(\epsilon_{xx},\epsilon_{yy},\epsilon_{zz},
2\epsilon_{yz},2\epsilon_{xz},2\epsilon_{xy}).
\]

不要对 stiffness coefficient 再额外乘或除 2/4。

### B3. Dielectric

```yaml
dielectric:
  electronic_field: dielectric
  ionic_field: dielectric_ionic
  unit: dimensionless_relative_permittivity
  primary_target: total_static
```

定义：

\[
\epsilon_{\mathrm{electronic}}^r
=
\texttt{dielectric},
\]

\[
\epsilon_{\mathrm{ionic}}^r
=
\texttt{dielectric\_ionic},
\]

\[
\epsilon_{\mathrm{static}}^r
=
\epsilon_{\mathrm{electronic}}^r
+
\epsilon_{\mathrm{ionic}}^r.
\]

用于势函数时：

\[
\chi_{\mathrm{static}}^r
=
\epsilon_{\mathrm{static}}^r-I.
\]

### B4. 更新 response audit

重新生成 audit，明确写入：

- 上述字段；
- 单位；
- conversion；
- tensor symmetry；
- 交集数量；
- 缺失标签比例；
- electronic/ionic/static 的数值范围。

保留原先的 `RESPONSE_DATA_BLOCKED.md` 作为历史记录，但新增：

```text
outputs/response_audit/RESOLVED.md
```

说明阻塞如何解除、依据什么配置继续。

---

## Phase C：先实现 M5 tensor utilities 和测试

不要立即开始完整多任务训练。

### C1. Elastic tensor converter

实现：

\[
C_{IJ}
\leftrightarrow
C_{ijkl},
\]

满足：

\[
C_{ijkl}
=
C_{jikl}
=
C_{ijlk}
=
C_{klij}.
\]

优先使用：

```python
e3nn.io.CartesianTensor("ijkl=ijlk=jikl=klij")
```

禁止手写 21 维 irreps 基变换。

### C2. Dielectric converter

实现：

\[
\epsilon_{ij}=\epsilon_{ji}.
\]

优先使用 `CartesianTensor`。

### C3. 必须新增测试

#### Elastic round trip

\[
C_{6\times6}
\rightarrow C_{3\times3\times3\times3}
\rightarrow \text{irreps}
\rightarrow C_{3\times3\times3\times3}
\rightarrow C_{6\times6}.
\]

#### Elastic symmetry

验证 minor/major symmetry。

#### Pure shear energy equivalence

随机 \(\gamma_{yz}\)，比较 Voigt 与 Cartesian：

\[
\frac12\eta_V^\mathsf TC^V\eta_V
=
\frac12\epsilon_{ij}C_{ijkl}\epsilon_{kl}.
\]

该测试非常重要，用于发现 shear factor 错误。

#### Dielectric round trip

\[
\epsilon_{3\times3}
\leftrightarrow \text{irreps}.
\]

#### Static dielectric

\[
\epsilon_{\mathrm{static}}^r
=
\epsilon_{\mathrm{electronic}}^r
+
\epsilon_{\mathrm{ionic}}^r.
\]

#### Susceptibility

\[
\chi^r=\epsilon^r-I.
\]

#### Rotation equivariance

elastic 和 dielectric 都要做随机 \(R\in O(3)\) 测试。

---

## Phase D：实现 M5A 的最小多响应模型

### D1. 模型结构

继续复用当前周期等变 encoder。

只增加小 head：

```text
piezo_head
elastic_head
dielectric_electronic_head
dielectric_ionic_head
```

不要复制 encoder。

### D2. 输出

- piezo：18 维 irreps；
- elastic：21 维物理自由度对应 irreps；
- dielectric electronic：6 维对称 tensor 对应 irreps；
- dielectric ionic：6 维对称 tensor 对应 irreps。

具体 irreps 维度从 `CartesianTensor` 读取，不手写字符串。

### D3. Missing-label mask

每个样本允许只具有部分标签。

\[
\mathcal L
=
\lambda_e\mathcal L_e
+
\lambda_C\mathcal L_C
+
\lambda_{\epsilon_e}\mathcal L_{\epsilon_e}
+
\lambda_{\epsilon_i}\mathcal L_{\epsilon_i}.
\]

每个 task：

\[
\mathcal L_t
=
\frac{
\sum_n m_{t,n}\ell_{t,n}
}{
\sum_n m_{t,n}+\epsilon
}.
\]

如果一个 batch 某任务没有标签：

- 该项为 0；
- 不产生 NaN；
- 不跳过整个 batch。

### D4. Normalization

每个 tensor family 独立使用：

- 一个 global RMS；或
- irrep-block RMS。

不得对 Cartesian 分量分别使用不同均值和标准差。

### D5. 数据读取

以 material ID 对齐。

不要：

- 把缺失标签填零；
- 用最近邻材料补标签；
- 猜测电子/离子介电；
- 只保留四任务完全交集。

---

## Phase E：M5A 小样本拟合

先做 32 个具有尽可能多响应标签的真实样本。

设置：

- batch size 32；
- weight decay 0；
- dropout 关闭；
- 300 epoch；
- seed 42；
- 同 cohort；
- 不称为 validation generalization。

目标：

- 四个 task 的有效 loss 都明显下降；
- 无 NaN；
- mask 计数正确；
- 各 head 都有非零梯度；
- 反归一化后的 tensor 数值有限；
- round-trip symmetry test 通过。

如果 32 个四任务交集不足：

- 允许使用 heterogeneous cohort；
- 每个 task 至少保证一定数量有效标签；
- 报告每个 task 有效样本数；
- 不生成假标签。

输出：

```text
outputs/m5/multitask_overfit/
├── best.pt
├── last.pt
├── metrics.csv
├── per_task_counts.csv
├── summary.json
└── report.md
```

---

## Phase F：M3 dry-run，而不是完整长训练

在完整 3-seed M3 前，先做一个端到端 dry-run。

### F1. Piezo-only M3 dry-run

使用完整 3998/500/500 数据管线，但只训练：

```text
2 epochs
seed 42
```

目的：

- 检查完整数据加载；
- 检查 validation；
- 检查 best checkpoint；
- 检查 resume；
- 检查 test-only evaluation；
- 检查 raw/projected metrics；
- 检查 aggregate 文件格式。

它不产生科学性能结论。

### F2. M5A dry-run

同样：

```text
2 epochs
seed 42
```

检查：

- heterogeneous batch；
- per-task mask；
- per-task validation；
- checkpoint；
- resume；
- evaluation。

### F3. Dry-run 验收

必须能够：

1. 中断后 resume；
2. best/last checkpoint 正确；
3. test 不参与 early stopping；
4. metrics JSON/CSV/Markdown 一致；
5. split 不被重写；
6. seed 被记录；
7. checkpoint 包含 normalization stats；
8. evaluation 可独立运行。

---

# 4. 完整 M3 什么时候启动

只有以下条件同时满足后再运行。

## 4.1 Piezo-only 配置冻结

- tensor convention；
- architecture；
- normalization；
- loss；
- projector；
- metrics；
- split；
- checkpoint resume。

## 4.2 M5A 配置冻结

- response convention；
- elastic/dielectric tests；
- mask；
- multiresponse overfit；
- dry-run。

## 4.3 决定最终需要跑的矩阵

不要盲目跑所有组合。

推荐之后只跑：

### Piezo-only baseline

```text
3 seeds
full loss
random split
```

### M5A shared multiresponse

```text
3 seeds
full loss
same piezo split
```

### OOD

先每个配置 1 seed：

```text
formula OOD
chemical-system OOD
```

如果结果正常，再补 3 seeds。

---

# 5. 暂时不要做的事项

本阶段不要：

- 开始 M5B field-conditioned scalar network；
- 跑 full/direct/JVP 各 3-seed 长训练；
- 扩大 sketch 超参搜索；
- 跑全部 OOD 的 3 seeds；
- 引入新的数据源；
- 重写 encoder；
- 引入 Lightning/Hydra；
- 为 GMTNet baseline 安装一大套新环境并阻塞主线。

GMTNet baseline 的 `wandb`/`dgl` 阻塞报告先保留。后续可创建独立环境解决，不要污染当前 EGNN 环境。

---

# 6. Commit 规划

每个阶段独立 commit：

```text
experiment: complete M4 gradient fidelity matrix
data: resolve response conventions and refresh audit
tensor: add elastic and dielectric conversions
test: validate shear energy and response tensor equivariance
model: add masked multiresponse coefficient heads
experiment: overfit heterogeneous multiresponse cohort
experiment: add end-to-end M3 dry runs
```

不要将所有功能放在一个巨大 commit。

---

# 7. 本轮验收条件

本轮完成的最低条件：

- M4 100-trial gradient fidelity 完成；
- 一个短周期 full/sketch 收敛对比完成；
- response conventions 固化为 YAML；
- M5 audit 标记 resolved；
- elastic converter 与 pure shear energy test 通过；
- dielectric electronic/ionic/static/susceptibility 逻辑通过；
- 多响应 masked heads 实现；
- M5A 32 样本或 heterogeneous cohort 可拟合；
- piezo-only 和 M5A 的 2-epoch full-data dry-run 通过；
- resume 和 test-only evaluation 通过；
- 没有启动耗时的完整 3-seed M3；
- 工作区 clean；
- 所有阻塞和边界如实记录。

---

# 8. 给 Codex 的第一步

先不要直接编码。

第一条回复只提交：

1. 当前与本任务有关的文件列表；
2. 准备修改的文件；
3. 准备新增的文件；
4. response convention 将放在哪个唯一配置文件；
5. elastic Voigt ↔ Cartesian 的实现方案；
6. dielectric electronic/ionic/static 的数据流；
7. multiresponse mask 的 batch 表示；
8. M4 收尾、M5A、M3 dry-run 的执行顺序；
9. 明确说明本轮不会运行完整 3-seed M3。

确认后再开始编码。
