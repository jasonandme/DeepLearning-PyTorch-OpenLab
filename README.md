# DeepLearning-PyTorch-OpenLab

## 项目简介

DeepLearning-PyTorch-OpenLab 是一个面向深度学习初学者的 PyTorch 学习实验室。

项目提供可运行的 Jupyter Notebook、简洁的实验代码、调试指南和概念问答，帮助学习者从 Tensor、Autograd、训练循环等基础内容，逐步过渡到 MLP、CNN、RNN/LSTM 和 Attention/Transformer 的入门实践。

项目重点不是提出新的算法，而是把基础概念、代码实现、实验观察和常见问题整理成一个可运行、可复现、可扩展的学习项目。

## 项目适合谁

- 正在入门深度学习和 PyTorch 的学习者。
- 想系统理解 Tensor、Autograd、Module、Loss、Optimizer 的使用方式。
- 希望通过小实验掌握线性回归、MLP、CNN、RNN/LSTM 和 Attention 基础。
- 想参考一个结构清晰、便于持续扩展的 PyTorch 学习项目。

## 项目目标

- 用简洁实验解释深度学习基础概念。
- 用 PyTorch 实现完整训练流程，包括数据、模型、损失、优化和评估。
- 通过 loss 曲线、预测结果和中间输出观察模型行为。
- 整理常见错误和排查方法，降低初学调试成本。
- 保持项目结构清晰，方便继续添加 Notebook、实验脚本和文档。

## 学习路线

建议按以下顺序学习：

1. PyTorch 基础：Tensor、Autograd、Module、Loss、Optimizer。
2. 第一个训练闭环：线性回归。
3. 非线性建模：MLP 分类。
4. 图像特征学习：CNN 图像分类。
5. 序列建模：RNN/LSTM。
6. 注意力机制：Attention 与 Transformer 基础。
7. 后续扩展：更多数据集、模型对比、训练日志、推理 Demo 和部署示例。

完整路线见 [docs/learning_path.md](docs/learning_path.md)。

## 项目结构

```text
DeepLearning-PyTorch-OpenLab/
├── README.md
├── requirements.txt
├── environment.yml
├── .gitignore
├── LICENSE
├── CONTRIBUTING.md
├── notebooks/
│   ├── 01_tensor_autograd_basics.ipynb
│   ├── 02_linear_regression_from_scratch.ipynb
│   ├── 03_mlp_classification.ipynb
│   ├── 04_cnn_image_classification.ipynb
│   ├── 05_rnn_lstm_sequence_modeling.ipynb
│   └── 06_attention_transformer_basics.ipynb
├── src/
│   ├── models/
│   ├── datasets/
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── docs/
│   ├── learning_path.md
│   ├── concept_qa.md
│   ├── debugging_guide.md
│   ├── experiment_notes.md
│   └── roadmap.md
├── examples/
│   ├── minimal_training_loop.py
│   ├── linear_regression_demo.py
│   └── mlp_demo.py
├── assets/
│   └── images/
└── results/
    └── README.md
```

## Notebook 目录

| 编号 | Notebook | 主题 | 主要内容 | 运行状态 |
| --- | --- | --- | --- | --- |
| 01 | `01_tensor_autograd_basics.ipynb` | Tensor 与 Autograd | Tensor 创建、shape、广播、矩阵运算、自动求导 | 可运行 |
| 02 | `02_linear_regression_from_scratch.ipynb` | 线性回归 | 手写参数、MSELoss、梯度下降、loss 曲线和拟合结果 | 可运行 |
| 03 | `03_mlp_classification.ipynb` | MLP 分类 | 非线性分类、ReLU、CrossEntropyLoss、决策边界观察 | 可运行 |
| 04 | `04_cnn_image_classification.ipynb` | CNN 图像分类 | 卷积、池化、digits 小图像分类、loss 曲线 | 可运行 |
| 05 | `05_rnn_lstm_sequence_modeling.ipynb` | RNN/LSTM 序列建模 | 滑动窗口、LSTM、下一步预测、预测曲线 | 可运行 |
| 06 | `06_attention_transformer_basics.ipynb` | Attention 与 Transformer 基础 | Q/K/V、注意力权重、简单 Transformer Block | 可运行 |

每个 Notebook 都包含本节目标、背景问题、核心概念、最小代码示例、完整实验、实验观察、常见错误、概念问答和小结。

## 快速开始

```bash
git clone https://github.com/your-username/DeepLearning-PyTorch-OpenLab.git
cd DeepLearning-PyTorch-OpenLab
pip install -r requirements.txt
jupyter notebook
```

如果使用 Conda：

```bash
conda env create -f environment.yml
conda activate dl-pytorch-openlab
jupyter notebook
```

## 示例运行

运行最小训练循环：

```bash
python examples/minimal_training_loop.py
```

运行线性回归示例：

```bash
python examples/linear_regression_demo.py
```

运行 MLP 分类示例：

```bash
python examples/mlp_demo.py
```

也可以使用 `src/train.py` 中的通用训练入口：

```bash
python -m src.train --task linear --epochs 80
python -m src.train --task mlp --epochs 80
```

## 文档说明

- [docs/learning_path.md](docs/learning_path.md)：面向初学者的学习路线，说明从 PyTorch 基础到 Attention 的学习顺序。
- [docs/concept_qa.md](docs/concept_qa.md)：概念问答，覆盖 PyTorch、梯度下降、线性回归、MLP、CNN、RNN/LSTM 和 Attention/Transformer。
- [docs/debugging_guide.md](docs/debugging_guide.md)：常见 PyTorch 错误与修正方式，包括 shape、dtype、学习率、训练/评估模式等问题。
- [docs/experiment_notes.md](docs/experiment_notes.md)：记录各实验的目标、设置、观察和结论。
- [docs/roadmap.md](docs/roadmap.md)：后续计划，包括更多数据集、模型对比、训练日志、推理 Demo 和部署示例。

## 项目亮点

- 从基础概念到代码实现，学习路径连续。
- 每个模块都有可运行示例，便于本地复现和修改。
- 训练实验包含 loss 曲线、预测结果或权重可视化，方便观察模型行为。
- 调试指南整理了初学 PyTorch 时常见的问题和最小代码示例。
- 文档、Notebook、源码和示例脚本分层组织，适合持续扩展。

## 当前进度

- [x] Tensor 与 Autograd 基础 Notebook。
- [x] 线性回归、MLP、CNN 基础实验。
- [x] RNN/LSTM 序列建模实验。
- [x] Attention 与 Transformer 基础实验。
- [x] PyTorch 概念问答。
- [x] 常见错误调试指南。
- [x] 示例训练脚本和通用训练入口。

## Roadmap

- [ ] 增加更多小型数据集实验。
- [ ] 增加模型结构和优化器对比。
- [ ] 增加训练日志记录与结果保存。
- [ ] 增加模型推理 Demo。
- [ ] 增加简单部署示例。
- [ ] 增加更完整的 Transformer 实现。

详细计划见 [docs/roadmap.md](docs/roadmap.md)。

## 如何贡献

欢迎提交 issue、文档改进、bug fix 或新的 Notebook。

建议贡献前先阅读 [CONTRIBUTING.md](CONTRIBUTING.md)，并尽量保持以下原则：

- Notebook 保持可运行，优先使用小型数据集或合成数据。
- 实验代码保持简洁、可读、可复现。
- 新增实验建议包含目标、实现、观察和常见错误说明。
- 不提交大型数据文件、模型权重、缓存文件或本地环境目录。

## License

本项目采用 [MIT License](LICENSE)。
