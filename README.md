# 工程实践 从公式实现一层 GCN

本工程从 GCN 数学公式实现单层图卷积，并以 DGL `GraphConv(norm="both")` 为参考，验证稠密邻接矩阵实现、边列表消息传递实现的输出与梯度。实验只使用 CPU、人工图和固定种子随机图，包含真实测试日志、数值表格、性能测量、可视化和中文 Word 报告。

## 核心公式

约定边 `j -> i` 对应 `A[i,j]`。默认严格执行 `A_tilde = A + I`：保留已有自环，再给每个节点增加一条自环。

```text
S = D_in^(-1/2) @ A_tilde @ D_out^(-1/2)
M = S @ H
Z = M @ W + b
H_out = activation(Z)
```

无向图的入度与出度相同；有向图按 DGL `norm="both"` 使用源节点出度和目标节点入度。零度节点的逆平方根置零。

## 项目结构

```text
src/          图预处理、ManualGCN、MessagePassingGCN、ReferenceGCN
tests/        正确性、边界条件与梯度一致性测试
experiments/  基线、人工样例、数值对照、性能、图表与报告脚本
data/         数据来源说明和实际图配置
results/      环境、日志、CSV、PNG 与汇总 JSON
report/       GCN工程实践实验报告.docx
run_all.py    一键执行入口
```

## 环境安装

本次实际验证使用 Python 3.10.20、PyTorch 2.5.1、DGL 1.1.2。Windows 下建议使用 Python 3.10 创建环境；DGL 1.1.2 是本次包索引中可用的兼容 wheel。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

若本机 PyTorch 需要纯 CPU wheel，可先按 PyTorch 官方 CPU 索引安装 `torch==2.5.1`，再安装其余依赖。所有实验张量都在 CPU 上创建，脚本会设置 `CUDA_VISIBLE_DEVICES` 为空。

## 运行方法

一键完成环境记录、DGL 基线、人工样例、Pytest、随机图对照、CPU 性能、可视化、汇总与 Word 报告：

```powershell
.\.venv\Scripts\python.exe run_all.py
```

单独运行：

```powershell
.\.venv\Scripts\python.exe experiments\run_baseline.py
.\.venv\Scripts\python.exe experiments\run_demo.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe experiments\run_comparison.py
.\.venv\Scripts\python.exe experiments\run_benchmark.py
.\.venv\Scripts\python.exe experiments\visualize.py
```

核心测试失败时 `run_all.py` 返回非零状态。固定随机种子为 `20261125`，PyTorch 线程数为 1。

## 边界规则与测试

- `add`：严格 `A + I`，保留原有自环并额外添加单位阵。
- `exactly_one`：先移除全部自环，再为每个节点添加一条。
- `none`：不添加自环，零度归一化系数为零。
- 普通重复边默认保留；可用 `deduplicate=True` 去重。
- 明确区分有节点无边图与零节点空图。
- 覆盖普通/随机无向图、有向图、孤立点、自环、重复边、单节点、零节点、特征扩张/压缩、偏置、激活、非法输入和梯度。

## 结果与交付物

- 环境：`results/environment.txt`
- 测试日志：`results/logs/pytest.log`
- 测试明细：`results/tables/test_results.csv`
- 数值误差：`results/tables/numerical_comparison.csv`
- 图策略：`results/tables/graph_strategy_comparison.csv`
- CPU 性能：`results/tables/benchmark.csv`
- 可视化：`results/figures/`
- 汇总：`results/experiment_summary.json`
- 报告：`report/GCN工程实践实验报告.docx`

性能表报告两类前向计算耗时，采用预热、多次重复、中位数与 IQR。测量未关闭自动微分，但不执行反向传播。预构建输入记录中，矩阵版复用归一化邻接矩阵，消息传递版复用边归一化系数，DGL 版复用已构建图但 `GraphConv` 仍执行内部度归一化，因此三者并非统一排除了完全相同的预处理；完整前向记录还包含各包装层的图处理。理论上稠密矩阵存储为 O(N²)，边列表为 O(E)，但项目不会把该理论差异描述为实测内存。

## 数据来源与许可

数据全部为项目人工构造或固定随机种子生成，不需要外部下载。该实验是算子正确性验证，不划分训练集、验证集和测试集。详细配置与许可说明见 `data/README.md` 和 `data/graph_cases.json`。
