# PiezoJet：将压电等耦合响应学习为单一势函数的等变 Differential Jet

> 本文档是给 Codex 的实现任务书。目标是先完成一个**可运行、可测试、结构简洁的研究原型**，而不是一次性堆出完整论文系统。  
> 请严格按里程碑执行；下载、数据格式、张量约定或依赖出现阻塞时，立即停止并汇报，不要静默更换数据集、生成假数据或增加多层回退代码。

---

## 0. 必须遵守的工程规则

1. **少文件、少抽象、少依赖**。不要创建 registry、plugin、factory 大体系，不要为了“未来扩展”写空壳类。
2. 优先使用现成库：
   - `torch`
   - `torch_geometric`
   - `e3nn`
   - `pymatgen`
   - `spglib`
   - `numpy`
   - `pyyaml`
   - `tqdm`
   - `pytest`
3. 训练循环使用普通 PyTorch；暂不引入 Lightning、Hydra、wandb。
4. 不复制第三方库源码。只调用公开 API。
5. 不使用宽泛的 `except Exception:`；错误应直接暴露，或抛出带上下文的明确异常。
6. 不写“自动尝试多个 URL / 多个数据集 / 多种格式”的回退逻辑。
7. 数据下载失败、文件字段不匹配、单位或 Voigt 约定不确定时：
   - 立即停止；
   - 不继续训练；
   - 输出一份简短阻塞报告，包含 URL、命令、完整错误、期望文件路径和需要用户提供的内容。
8. 禁止用随机张量冒充真实数据完成训练。随机数据只允许用于单元测试。
9. 所有实验必须可复现：固定 seed，保存配置、split 文件、归一化常数和 Git commit。
10. 保持函数短小。能用 `einsum`、`e3nn.io.CartesianTensor`、PyG scatter 完成的，不手写冗长索引循环。

---

## 1. 研究目标

实现一个晶体响应模型，核心学习对象不是孤立的压电张量 head，而是一个标量响应势函数

\[
\Phi_\theta(x,\boldsymbol{\eta},\mathbf E),
\]

其中：

- \(x=(Z,L,F)\) 是晶体；
- \(Z\) 是原子种类；
- \(L\) 是晶格；
- \(F\) 是分数坐标；
- \(\boldsymbol{\eta}\) 是对称应变；
- \(\mathbf E\) 是外电场。

压电张量由混合导数得到：

\[
\widehat e_{\alpha\mu}
=
-\left.
\frac{\partial^2\Phi_\theta}
{\partial E_\alpha\partial\eta_\mu}
\right|_{\mathbf E=0,\boldsymbol{\eta}=0}.
\]

第一版先实现**完整三阶压电张量**。第二阶段再加入介电与弹性张量，使

\[
\Phi_\theta
=
\frac12\eta^\mathsf T \widehat C\,\eta
-
E^\mathsf T\widehat e\,\eta
-
\frac12 E^\mathsf T\widehat\chi\,E.
\]

这样可自动保证：

\[
\widehat C=\widehat C^\mathsf T,\qquad
\widehat\chi=\widehat\chi^\mathsf T,
\]

以及混合导数交换：

\[
\frac{\partial^2\Phi}
{\partial E_\alpha\partial\eta_\mu}
=
\frac{\partial^2\Phi}
{\partial\eta_\mu\partial E_\alpha}.
\]

### 第一版明确不做

- 不做 DFT；
- 不做 relaxed-ion 隐式微分；
- 不做 Berry-phase 分支展开；
- 不做材料生成；
- 不同时支持多个张量数据源；
- 不实现大型 foundation model。

这些属于第二轮研究扩展，不应阻塞 MVP。

---

## 2. 唯一指定的数据源

### 2.1 主数据集

使用 GMTNet 官方仓库公开的 JARVIS-DFT 张量数据：

- 仓库：`https://github.com/YKQ98/GMTNet`
- 数据目录：仓库中的 `/data`
- 论文代码说明该目录包含 dielectric、piezoelectric、elastic tensor 数据，并强调结构和张量来自一致的 DFT 计算文件。

第一阶段只读取 piezoelectric 数据；第二阶段才读取 dielectric 和 elastic。

### 2.2 下载行为

只实现一个下载脚本：

```bash
python scripts/download_data.py --output data/raw/gmtnet
```

建议行为：

- 使用一次明确的 `git clone --depth 1 https://github.com/YKQ98/GMTNet.git ...`；
- clone 成功后检查 `/data` 是否存在；
- 记录仓库 commit SHA；
- 不自动下载 JARVIS 的另一个版本；
- 不自动切换到 Materials Project；
- 不从论文 PDF OCR 数据。

若 clone 或文件检查失败，停止并生成：

```text
DOWNLOAD_BLOCKED.md
```

模板见本文末尾。

### 2.3 数据检查必须先于建模

实现：

```bash
python scripts/inspect_data.py --root data/raw/gmtnet
```

输出至少包括：

- 文件名；
- 样本数；
- 第一条记录的字段名；
- structure 的表示形式；
- tensor shape；
- tensor 单位；
- Voigt 顺序；
- shear 是否使用 engineering strain；
- 是否已有 train/val/test split；
- 空值、NaN、Inf 数量；
- 每个样本的原子数范围；
- centrosymmetric 样本中张量是否接近零。

**不得猜测字段名、单位或 Voigt 约定。**  
以 GMTNet 数据 loader 和仓库代码为准。若无法从代码与数据唯一确定，停止并汇报。

### 2.4 Split

优先使用仓库已有 split。若确实没有：

- seed = 42；
- 80/10/10；
- 按 material id 排序后再随机；
- 把结果写入 `data/processed/splits.json`；
- 后续运行只能读取该文件，不要每次重分。

可增加一个 OOD split，但不能阻塞 MVP：

- 按 reduced formula 分组；
- 同一 formula 不跨 split；
- 文件名 `splits_formula_ood.json`。

---

## 3. 张量与表示论约定

### 3.1 应变

使用 6 维 Voigt 输入：

\[
\eta_6=
(\eta_{xx},\eta_{yy},\eta_{zz},
\eta_{yz}^{V},\eta_{xz}^{V},\eta_{xy}^{V}).
\]

具体 off-diagonal 是否包含因子 2，必须由数据检查确定，并集中写在一个函数中：

```python
voigt_to_symmetric_matrix(eta6, convention)
symmetric_matrix_to_voigt(eta, convention)
```

整个项目只能有这一份约定实现。

### 3.2 压电张量

数据通常为 \(3\times6\)，内部统一转为

\[
e_{ijk}=e_{ikj},
\]

即 Cartesian shape `[3, 3, 3]`。

使用：

```python
from e3nn.io import CartesianTensor
piezo_type = CartesianTensor("ijk=ikj")
```

进行 Cartesian tensor 与 irreps 之间的转换，不手写 Clebsch–Gordan 基变换。

应验证：

\[
\mathrm{Piez}
\cong
2\mathcal H^1_o
\oplus
\mathcal H^2_o
\oplus
\mathcal H^3_o,
\]

总维度：

\[
2\cdot 3+5+7=18.
\]

不要把 irreps 字符串硬编码为唯一真值；程序启动时从 `CartesianTensor` 读取并 assert 总维度为 18。

### 3.3 联合旋转

模型应满足：

\[
\widehat e(R\cdot x)
=
\rho_3(R)\widehat e(x),
\]

其中

\[
[\rho_3(R)e]_{ijk}
=
R_{ia}R_{jb}R_{kc}e_{abc}.
\]

标量势满足：

\[
\Phi_\theta
(R\cdot x,R\eta R^\mathsf T,R E)
=
\Phi_\theta(x,\eta,E).
\]

单元测试必须验证这一点。

---

## 4. MVP 模型

### 4.1 总体结构

实现三个清晰模块，不要再细分十几个文件：

1. `PeriodicCrystalEncoder`
2. `PiezoTensorHead`
3. `ResponsePotential`

数据流：

```text
crystal
  -> periodic equivariant graph encoder
  -> graph-level piezo irreps
  -> Cartesian piezo tensor
  -> scalar response potential Phi(E, eta)
  -> mixed derivative / tensor prediction
```

### 4.2 Periodic graph

- 原子节点：atomic number embedding，仅为 scalar irreps；
- 边：使用 cutoff radius，保留 PBC image shift；
- Cartesian edge vector：

\[
r_{ij}= (f_j-f_i+n_{ij})L;
\]

- spherical harmonics 到 `lmax=3`；
- 3 个 message-passing block 足够；
- 默认 cutoff 5 Å，max neighbors 32；
- 使用 PyG batch；
- 使用 `torch_geometric.utils.scatter` 或 PyTorch `scatter_reduce`，不要额外手写 scatter。

建议 hidden irreps 保持小型：

```text
64x0e + 16x0o
+ 24x1e + 24x1o
+ 12x2e + 12x2o
+ 6x3e + 6x3o
```

若显存不足，统一减半，不要引入动态架构搜索。

每个 message block：

- radial basis；
- radial MLP 产生 tensor-product 权重；
- `e3nn.o3.FullyConnectedTensorProduct`；
- edge aggregation；
- `e3nn.nn.Gate`；
- residual connection。

### 4.3 Graph readout

对 node irreps 做 graph-wise sum 或 mean，随后用 `e3nn.o3.Linear` 输出：

```python
irreps_out = piezo_type
```

得到 18 维 piezo irreps，再用：

```python
piezo_type.to_cartesian(...)
```

变为 `[B, 3, 3, 3]`。

### 4.4 ResponsePotential

第一版使用明确的双线性响应势：

\[
\Phi_\theta(x,E,\eta)
=
- E_\alpha\,
\widehat e_{\alpha jk}(x)\,
\eta_{jk}.
\]

实现应接收 `eta6`，内部只通过统一的 Voigt 转换函数得到 \(\eta_{jk}\)。

核心代码应接近：

```python
phi = -torch.einsum("bi,bijk,bjk->b", field, piezo_cart, strain)
```

不要在第一版乘体积 \(\Omega\)。当前训练目标是张量本身，单位问题待有 electric enthalpy 标签后再处理。

### 4.5 归一化

为了不破坏等变性：

- 不对 18 个 Cartesian 分量分别减均值；
- 第一版只使用一个全局正标量：

\[
s_e=
\sqrt{
\frac{1}{18N}
\sum_n \|e_n\|_F^2
}.
\]

训练目标为 \(e/s_e\)，评测再乘回 \(s_e\)。

保存为：

```text
data/processed/stats.json
```

---

## 5. 随机混合 Hessian Sketch

### 5.1 数学目标

随机采样：

\[
a\sim\mathcal N(0,I_3),\qquad
b\sim\mathcal N(0,I_6).
\]

模型方向混合导数：

\[
s_\theta(a,b)
=
-
\left.
\frac{\partial^2}
{\partial t\,\partial r}
\Phi_\theta(x,rb,ta)
\right|_{r=t=0}.
\]

目标：

\[
s^\star(a,b)=a^\mathsf T e^\star b.
\]

损失：

\[
\mathcal L_{\mathrm{sketch}}
=
\mathbb E_{a,b}
\left[
s_\theta(a,b)-a^\mathsf T e^\star b
\right]^2.
\]

有：

\[
\mathbb E_{a,b}
(a^\mathsf T\Delta e\,b)^2
=
\|\Delta e\|_F^2.
\]

### 5.2 实现要求

使用 `torch.func.jvp` 的嵌套 JVP。不要显式构建每个样本的完整 18 元 Hessian用于训练。

伪代码：

```python
def phi_of_eta(eta6):
    return model.potential(batch, field, eta6)

def directional_eta(field):
    _, d_eta = torch.func.jvp(
        lambda eta6: model.potential(batch, field, eta6),
        (eta0,),
        (b,),
    )
    return d_eta

_, mixed = torch.func.jvp(
    directional_eta,
    (field0,),
    (a,),
)
prediction = -mixed
```

若 `torch.func.jvp` 与当前 graph op 不兼容，不要改写成复杂的自定义 autograd。先：

1. 写一个最小复现；
2. 定位具体不兼容 op；
3. 优先替换为 PyTorch 原生等价 op；
4. 仍无法解决则停止并汇报。

### 5.3 训练模式

CLI 提供：

```bash
--loss full
--loss sketch
--loss hybrid
```

- `full`：完整 tensor MSE；
- `sketch`：每样本每步 1 个随机投影；
- `hybrid`：sketch + 小权重 full loss。

默认先跑 `full` 验证网络，再跑 `sketch`。

---

## 6. 第二阶段：统一介电、弹性和压电响应

只有第一阶段测试通过后才做。

标量势：

\[
\Phi_\theta
=
\frac12\eta^\mathsf T C\eta
-
E^\mathsf T e\eta
-
\frac12E^\mathsf T\chi E.
\]

由自动微分恢复：

\[
C_{\mu\nu}
=
\frac{\partial^2\Phi}
{\partial\eta_\mu\partial\eta_\nu},
\]

\[
\chi_{\alpha\beta}
=
-\frac{\partial^2\Phi}
{\partial E_\alpha\partial E_\beta},
\]

\[
e_{\alpha\mu}
=
-\frac{\partial^2\Phi}
{\partial E_\alpha\partial\eta_\mu}.
\]

要求：

- 同一 encoder；
- 三个 response coefficient head；
- 同一个 `ResponsePotential`；
- 缺失标签使用显式 mask；
- 不复制三套训练代码；
- 介电和弹性数据仍只来自 GMTNet `/data`。

第二阶段增加 Maxwell gap：

\[
\mathrm{MaxwellGap}
=
\left\|
\frac{\partial^2\Phi}{\partial E\,\partial\eta}
-
\left(
\frac{\partial^2\Phi}{\partial\eta\,\partial E}
\right)^\mathsf T
\right\|_F.
\]

---

## 7. 最小项目结构

```text
piezojet/
├── pyproject.toml
├── README.md
├── config.yaml
├── scripts/
│   ├── download_data.py
│   └── inspect_data.py
├── src/piezojet/
│   ├── data.py
│   ├── tensor_ops.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
└── tests/
    ├── test_tensor_ops.py
    ├── test_equivariance.py
    └── test_sketch.py
```

不要再增加 `utils/`, `core/`, `common/`, `helpers/`, `registry/` 等目录。

### 文件职责

- `data.py`：读取、split、PBC graph；
- `tensor_ops.py`：Voigt、旋转、CartesianTensor、归一化；
- `model.py`：encoder、head、potential；
- `train.py`：训练入口与 checkpoint；
- `evaluate.py`：指标和等变性评测；
- `config.yaml`：单一配置文件。

---

## 8. 必须写的测试

### 8.1 Voigt round trip

\[
\eta_6
\rightarrow
\eta_{3\times3}
\rightarrow
\eta_6
\]

误差小于 `1e-7`。

### 8.2 Piezo Cartesian round trip

\[
e_{3\times6}
\rightarrow
e_{3\times3\times3}
\rightarrow
\text{irreps}
\rightarrow
e_{3\times3\times3}
\rightarrow
e_{3\times6}.
\]

误差小于 `1e-5`。

### 8.3 旋转等变

随机 \(R\in O(3)\)：

\[
\frac{
\|\widehat e(Rx)-\rho_3(R)\widehat e(x)\|_F
}{
\|\widehat e(x)\|_F+\epsilon
}
<10^{-4}
\]

在未训练随机权重下也应成立。

### 8.4 势函数不变

\[
|\Phi(Rx,R\eta R^\mathsf T,RE)-\Phi(x,\eta,E)|<10^{-5}.
\]

### 8.5 Sketch 无偏数值验证

对随机已知矩阵 \(e\)，Monte Carlo 平均验证：

\[
\mathbb E(a^\mathsf T e b)^2
\approx
\|e\|_F^2.
\]

### 8.6 混合导数交换

分别按 \(E\rightarrow\eta\) 和 \(\eta\rightarrow E\) 做 JVP，误差小于 `1e-5`。

---

## 9. 训练和评测命令

```bash
python scripts/download_data.py --output data/raw/gmtnet
python scripts/inspect_data.py --root data/raw/gmtnet
pytest -q

python -m piezojet.train \
  --config config.yaml \
  --loss full

python -m piezojet.train \
  --config config.yaml \
  --loss sketch

python -m piezojet.evaluate \
  --checkpoint outputs/best.pt \
  --split test
```

训练输出只需：

```text
outputs/
├── best.pt
├── last.pt
├── config.resolved.yaml
├── metrics.csv
└── summary.json
```

不要保存每个 epoch 的 checkpoint。

---

## 10. 评测指标

至少报告：

1. Cartesian component MAE；
2. sample-wise Frobenius MAE；
3. normalized Frobenius error；
4. \(\max |e_{ij}|\) MAE；
5. rotation equivariance residual；
6. point-group residual，可用 `pymatgen`/`spglib` 对称操作；
7. sketch 与 full loss 的显存、时间、精度比较；
8. random split 和 formula-OOD split；
9. centrosymmetric false-positive norm。

不要只报告一个平均 MAE。

---

## 11. 里程碑和停止点

### M0：环境和数据

- 安装依赖；
- 下载 GMTNet；
- inspect schema；
- 确认单位和 Voigt。

**任何一项失败：停止。**

### M1：张量工具

- 所有 tensor tests 通过；
- 不开始训练前先完成 rotation test。

### M2：小数据过拟合

- 取 32 个真实样本；
- full tensor loss；
- 能显著过拟合；
- 无 NaN。

### M3：完整 baseline

- full loss；
- random split；
- 保存结果。

### M4：Hessian sketch

- sketch test；
- 训练；
- 对比显存与速度。

### M5：多响应 Jet

- 加介电和弹性；
- 缺失标签 mask；
- Maxwell gap。

不要跨过失败里程碑继续堆功能。

---

## 12. 验收标准

MVP 完成需同时满足：

- 数据真实来自 GMTNet；
- tensor convention 有明确测试；
- 模型严格 O(3) 等变；
- scalar potential 在联合旋转下不变；
- full loss 可训练；
- sketch loss 可训练；
- mixed derivative order 数值一致；
- 32 样本可以过拟合；
- 项目结构不超过本文列出的主体文件；
- README 能从零复现实验；
- 没有静默 fallback。

---

## 13. 阻塞报告模板

当下载或数据不明确时，创建 `DOWNLOAD_BLOCKED.md`，然后停止：

```markdown
# Implementation blocked

## Step
例如：下载 GMTNet 数据

## Command
完整命令

## Source
完整 URL

## Error
原样粘贴错误

## Expected local path
例如：data/raw/gmtnet/data/

## What I checked
列出已检查的文件或仓库说明

## What is needed from the user
明确写出需要用户手动下载并放置的文件名和路径
```

不要在阻塞后生成假数据或改用别的数据集。

---

## 14. 参考资料

- GMTNet official repository: https://github.com/YKQ98/GMTNet
- GMTNet paper: https://proceedings.mlr.press/v235/yan24d.html
- JARVIS-Tools database access: https://jarvis-tools.readthedocs.io/en/master/databases.html
- JARVIS raw DFPT data description: https://jarvis-materials-design.github.io/dbdocs/thedownloads/
- e3nn documentation: https://docs.e3nn.org/en/stable/
- e3nn CartesianTensor: https://docs.e3nn.org/en/stable/api/io/cartesian_tensor.html
- e3nn TensorProduct: https://docs.e3nn.org/en/stable/api/o3/o3_tp.html
