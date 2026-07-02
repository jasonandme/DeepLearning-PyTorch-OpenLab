# 深度学习与 PyTorch 学习路线

这份路线对应 `notebooks/` 下的 18 个主题 Notebook。建议按顺序学习，因为后面的实验会复用前面的 Tensor、Autograd、训练循环和 shape 管理经验。

## 阶段一：PyTorch 基础

### 01 Tensor 基础

先学 Tensor，是因为 PyTorch 中的数据、参数、梯度和中间特征都用 Tensor 表示。需要重点掌握 shape、dtype、device、广播和矩阵乘法。

对应 Notebook：`01_tensor_basics.ipynb`

### 02 Autograd 与计算图

深度学习训练依赖梯度。Autograd 会根据 Tensor 运算构建计算图，并在 `backward()` 时计算梯度。需要理解梯度累积、清零和 `torch.no_grad()`。

对应 Notebook：`02_autograd_computation_graph.ipynb`

### 03 Module 与训练循环

`nn.Module` 用来组织模型结构和参数。标准训练循环包括前向计算、loss、清梯度、反向传播和优化器更新。

对应 Notebook：`03_nn_module_training_loop.ipynb`

### 04 Dataset 与 DataLoader

DataLoader 让训练可以按 mini-batch 进行，也能处理 shuffle、batch size 等训练细节。

对应 Notebook：`04_dataset_dataloader.ipynb`

## 阶段二：基础监督学习实验

### 05 线性回归

线性回归适合作为第一个完整训练闭环。它模型简单，但包含数据、模型、loss、优化和可视化观察。

对应 Notebook：`05_linear_regression.ipynb`

### 06 Logistic Regression 分类

分类任务需要理解 logits、类别标签和 `CrossEntropyLoss`。这一章帮助区分回归和分类的训练方式。

对应 Notebook：`06_logistic_regression_classification.ipynb`

### 07 MLP 深层网络

MLP 是理解非线性建模的关键。激活函数让模型能够学习弯曲的决策边界。

对应 Notebook：`07_mlp_deep_network.ipynb`

### 08 优化器与学习率

学习率会直接影响 loss 曲线。过大可能震荡，过小可能收敛慢。SGD 和 Adam 的对比能帮助理解优化器行为。

对应 Notebook：`08_optimization_learning_rate.ipynb`

### 09 正则化

当模型容量较大、数据较少时容易过拟合。Dropout 和 weight decay 是常见的正则化手段。

对应 Notebook：`09_regularization_dropout_weight_decay.ipynb`

## 阶段三：图像建模

### 10 CNN 基础

CNN 解决了全连接网络处理图像时弱化空间结构、参数量较大的问题。卷积层通过局部连接和参数共享学习图像模式。

对应 Notebook：`10_cnn_basics.ipynb`

### 11 CNN 图像分类

在 digits 数据集上训练小型 CNN，观察 loss、预测样例和分类效果。

对应 Notebook：`11_cnn_image_classification_digits.ipynb`

### 12 CNN 特征可视化

通过中间特征图和错误样本分析，理解模型不只是输出准确率，也可以被观察和复盘。

对应 Notebook：`12_cnn_feature_visualization.ipynb`

## 阶段四：序列建模

### 13 RNN 基础

RNN 适合处理有顺序关系的数据。重点理解 `[batch, seq_len, features]`、隐状态和最后时间步输出。

对应 Notebook：`13_rnn_sequence_basics.ipynb`

### 14 LSTM 时间序列预测

LSTM 通过门控机制更稳定地保留历史信息。滑动窗口实验可以帮助理解序列预测样本如何构造。

对应 Notebook：`14_lstm_time_series_prediction.ipynb`

## 阶段五：Attention 与 Transformer

### 15 从零实现 Attention

Attention 通过 Query、Key、Value 计算相关性权重，再加权汇总信息。它改变了序列建模中信息交互的方式。

对应 Notebook：`15_attention_from_scratch.ipynb`

### 16 Multi-Head Attention

多头注意力让模型从多个子空间观察关系。需要理解 `embed_dim`、`num_heads`、mask 和输出 shape。

对应 Notebook：`16_multi_head_attention.ipynb`

### 17 Transformer Encoder

Encoder Block 组合了多头注意力、残差连接、LayerNorm 和前馈网络，是理解 Transformer 的关键模块。

对应 Notebook：`17_transformer_encoder_basics.ipynb`

## 阶段六：训练工程

### 18 日志、Checkpoint 与复现实验

一个学习项目不仅要能训练模型，还应能记录 loss/accuracy、保存 checkpoint，并说明实验设置和结果。

对应 Notebook：`18_training_engineering_checkpoints_logs.ipynb`

## 推荐学习方式

1. 先完整运行 Notebook，不急着改代码。
2. 第二遍只改一个变量，例如学习率、隐藏层宽度或 batch size。
3. 记录 loss、accuracy 或可视化结果的变化。
4. 遇到报错先查 shape、dtype、loss 使用方式和训练/评估模式。
5. 学完一个阶段后，回到 `docs/concept_qa.md` 做概念复盘。
