# DeepLearning-PyTorch-OpenLab

## 项目简介

DeepLearning-PyTorch-OpenLab 是一个面向深度学习初学者的 PyTorch 开源学习实验室。

项目通过可运行的 Jupyter Notebook、简洁实验代码、调试指南、概念问答和实验记录，帮助学习者从 Tensor、Autograd、训练循环等基础内容，逐步过渡到 MLP、CNN、RNN/LSTM、Attention、Transformer 和训练工程实践。

项目定位是“基础能力学习与实践”，不是原创算法项目。内容强调可运行、可复现、可观察、可扩展。

如果这个项目对你的学习有帮助，欢迎 Star 支持，也欢迎通过 issue 提出改进建议。

## 项目适合谁

- 正在入门深度学习和 PyTorch 的学习者。
- 希望系统理解 Tensor、Autograd、Module、Loss、Optimizer 和 DataLoader 的使用方式。
- 想通过小实验掌握回归、分类、MLP、CNN、RNN/LSTM、Attention 和 Transformer 基础。
- 想参考一个结构清晰、便于持续扩展的 PyTorch 学习项目。

## 项目目标

- 用原创实验解释深度学习和 PyTorch 基础概念。
- 用可运行 Notebook 串联完整学习路线。
- 通过 loss 曲线、accuracy、预测结果和可视化结果观察模型行为。
- 整理常见错误和排查方法，降低初学调试成本。
- 保持项目结构清晰，方便继续添加实验、文档和示例代码。

## 学习路线

建议按以下顺序学习：

1. Tensor 基础与张量操作。
2. Autograd 与计算图。
3. `nn.Module` 与标准训练循环。
4. Dataset 与 DataLoader。
5. 线性回归完整训练闭环。
6. Logistic Regression 分类。
7. MLP 深层网络与非线性建模。
8. 优化器与学习率。
9. 正则化、Dropout 与 Weight Decay。
10. CNN 基础：卷积、池化与特征图。
11. CNN 图像分类实验。
12. CNN 特征可视化与错误分析。
13. RNN 序列建模基础。
14. LSTM 时间序列预测。
15. 从零实现 Attention。
16. Multi-Head Attention。
17. Transformer Encoder 基础。
18. 训练工程：日志、Checkpoint 与复现实验。

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
├── src/
├── docs/
├── examples/
├── assets/
└── results/
```

## Notebook 目录

| 编号 | Notebook | 主题 | 主要内容 | 运行状态 |
| --- | --- | --- | --- | --- |
| 01 | `01_tensor_basics.ipynb` | Tensor 基础 | Tensor 创建、属性、shape、广播、矩阵乘法 | 可运行 |
| 02 | `02_autograd_computation_graph.ipynb` | Autograd | 自动求导、计算图、梯度累积、最小参数更新 | 可运行 |
| 03 | `03_nn_module_training_loop.ipynb` | Module 与训练循环 | `nn.Module`、参数管理、标准训练流程 | 可运行 |
| 04 | `04_dataset_dataloader.ipynb` | Dataset/DataLoader | TensorDataset、DataLoader、mini-batch 训练 | 可运行 |
| 05 | `05_linear_regression.ipynb` | 线性回归 | 连续值预测、MSELoss、拟合曲线、loss 曲线 | 可运行 |
| 06 | `06_logistic_regression_classification.ipynb` | 线性分类 | logits、CrossEntropyLoss、分类准确率 | 可运行 |
| 07 | `07_mlp_deep_network.ipynb` | MLP | 非线性分类、ReLU、决策区域可视化 | 可运行 |
| 08 | `08_optimization_learning_rate.ipynb` | 优化器 | SGD、Adam、学习率对 loss 的影响 | 可运行 |
| 09 | `09_regularization_dropout_weight_decay.ipynb` | 正则化 | Dropout、weight decay、过拟合观察 | 可运行 |
| 10 | `10_cnn_basics.ipynb` | CNN 基础 | 卷积、池化、特征图 shape 和可视化 | 可运行 |
| 11 | `11_cnn_image_classification_digits.ipynb` | CNN 分类 | digits 图像分类、loss 曲线、预测样例 | 可运行 |
| 12 | `12_cnn_feature_visualization.ipynb` | CNN 观察 | 中间特征图、错误样本分析 | 可运行 |
| 13 | `13_rnn_sequence_basics.ipynb` | RNN 基础 | 序列输入、隐状态、最后时间步输出 | 可运行 |
| 14 | `14_lstm_time_series_prediction.ipynb` | LSTM | 滑动窗口、时间序列预测、预测曲线 | 可运行 |
| 15 | `15_attention_from_scratch.ipynb` | Attention | Q/K/V、scaled dot-product attention、权重可视化 | 可运行 |
| 16 | `16_multi_head_attention.ipynb` | 多头注意力 | `nn.MultiheadAttention`、mask、权重观察 | 可运行 |
| 17 | `17_transformer_encoder_basics.ipynb` | Transformer Encoder | Encoder Block、残差连接、LayerNorm、位置编码 | 可运行 |
| 18 | `18_training_engineering_checkpoints_logs.ipynb` | 训练工程 | 日志记录、checkpoint 保存与加载、复现实验 | 可运行 |

每个 Notebook 都包含：本节目标、背景问题、核心概念、最小代码示例、完整实验、实验观察、常见错误、概念问答、延伸练习和小结。

## 快速开始

```bash
git clone https://github.com/jasonandme/DeepLearning-PyTorch-OpenLab.git
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

```bash
python examples/minimal_training_loop.py
python examples/linear_regression_demo.py
python examples/mlp_demo.py
```

也可以使用通用训练入口：

```bash
python -m src.train --task linear --epochs 80
python -m src.train --task mlp --epochs 80
```

## 文档说明

- [docs/learning_path.md](docs/learning_path.md)：完整学习路线。
- [docs/concept_qa.md](docs/concept_qa.md)：深度学习与 PyTorch 概念问答。
- [docs/debugging_guide.md](docs/debugging_guide.md)：PyTorch 常见错误排查指南。
- [docs/experiment_notes.md](docs/experiment_notes.md)：实验目标、设置、观察和可改进点。
- [docs/roadmap.md](docs/roadmap.md)：后续维护和扩展计划。

## 项目亮点

- 从 PyTorch 基础到 Transformer Encoder，形成完整入门实践路线。
- 每个主题都有可运行 Notebook，而不是只给概念描述。
- 多数实验包含 loss、accuracy、预测结果或权重可视化。
- 包含调试说明、常见错误和延伸练习，适合边运行边修改。
- 文档、Notebook、源码、示例脚本分层组织，便于持续扩展。

## 当前进度

- [x] 18 个主题 Notebook。
- [x] Tensor、Autograd、Module、DataLoader 基础。
- [x] 回归、分类、MLP、优化和正则化实验。
- [x] CNN、RNN/LSTM、Attention、Transformer 基础实验。
- [x] 训练日志和 checkpoint 示例。
- [x] 学习路线、概念问答、调试指南和实验记录。

## Roadmap

- [ ] 增加 Fashion-MNIST / CIFAR-10 子集实验。
- [ ] 增加 MLP、CNN、RNN、LSTM、Transformer 的对比实验。
- [ ] 增加更完整的训练配置管理。
- [ ] 增加推理 Demo 和简单部署示例。
- [ ] 增加自动化测试，检查 Notebook 关键代码可运行。

详细计划见 [docs/roadmap.md](docs/roadmap.md)。

## 如何贡献

欢迎提交 issue、文档改进、bug fix 或新的 Notebook。建议贡献前阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。

贡献建议：

- Notebook 保持可运行，优先使用小型数据集或合成数据。
- 新增实验建议包含目标、实现、观察、常见错误和延伸练习。
- 不提交大型数据文件、模型权重、缓存文件或本地环境目录。

## 资源说明

本仓库只包含原创学习笔记、实验代码和项目文档，不分发第三方受版权保护材料。建议读者通过正规渠道获取拓展阅读材料。

## License

本项目采用 [MIT License](LICENSE)。
