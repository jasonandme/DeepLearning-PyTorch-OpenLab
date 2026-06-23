# 深度学习与 PyTorch 学习路线

这份路线面向刚开始学习深度学习和 PyTorch 的读者。建议先用小实验跑通核心流程，再逐步扩展到更复杂的模型和任务。

## 为什么先学 Tensor 和 Autograd

Tensor 是 PyTorch 中最基础的数据结构。输入数据、模型参数、梯度和中间特征都可以用 Tensor 表示。初学阶段如果不熟悉 Tensor 的 shape、dtype、广播和矩阵运算，后面写模型时很容易遇到维度错误。

Autograd 是 PyTorch 的自动求导机制。深度学习训练依赖梯度更新参数，而 Autograd 会根据 Tensor 运算自动构建计算图，并在 `backward()` 时计算梯度。

建议先掌握：

- 如何创建 Tensor。
- 如何查看和调整 shape。
- 什么是 `requires_grad`。
- `loss.backward()` 后梯度保存在哪里。
- 为什么训练循环中要清空梯度。

对应 Notebook：`01_tensor_autograd_basics.ipynb`。

## 为什么线性回归适合作为第一个训练闭环

线性回归模型简单，但训练流程完整。它能帮助学习者在较低认知负担下理解训练闭环：

```text
准备数据 -> 前向预测 -> 计算损失 -> 反向传播 -> 更新参数 -> 观察结果
```

线性回归适合入门的原因：

- 预测公式直观，通常写成 `y = wx + b`。
- 目标是连续值，方便用 MSELoss 衡量误差。
- loss 曲线和拟合直线都容易观察。
- 参数数量少，便于理解梯度更新。

对应 Notebook：`02_linear_regression_from_scratch.ipynb`。

## 为什么 MLP 是理解非线性建模的关键

线性模型只能表达线性关系。现实任务中，很多数据的边界不是直线。MLP 在多层线性变换之间加入激活函数，使模型具备非线性表达能力。

学习 MLP 时建议关注：

- `nn.Linear` 如何改变特征维度。
- ReLU 等激活函数为什么重要。
- 分类任务中 logits 和 label 如何传入 `CrossEntropyLoss`。
- 训练 loss 和测试 accuracy 如何一起观察。

MLP 是从“简单训练循环”过渡到“神经网络建模”的关键一步。

对应 Notebook：`03_mlp_classification.ipynb`。

## CNN 解决了全连接网络处理图像时的什么问题

如果直接把图像 flatten 后交给全连接网络，模型会弱化图像的空间结构，并且参数量容易变大。CNN 使用卷积核在图像局部区域滑动，通过参数共享学习局部特征。

CNN 的主要优势：

- 保留图像的二维空间结构。
- 利用局部连接学习边缘、形状等模式。
- 通过参数共享减少参数量。
- 通过池化降低特征图尺寸。

学习 CNN 时，最重要的是理解输入 shape：`[batch, channel, height, width]`。

对应 Notebook：`04_cnn_image_classification.ipynb`。

## RNN/LSTM 为什么适合序列数据

序列数据的特点是顺序重要。例如时间序列、文本和语音中，当前元素通常和前面元素有关。RNN 和 LSTM 会按时间步读取输入，并用隐状态保存历史信息。

普通 RNN 可以建模短期依赖，但在较长序列上容易训练困难。LSTM 引入门控机制，用更稳定的方式保留或遗忘信息，因此更适合处理较长的依赖关系。

学习序列模型时建议关注：

- 如何用滑动窗口构造监督学习样本。
- 输入 shape 为什么通常是 `[batch, seq_len, features]`。
- 为什么可以用最后一个时间步的输出做预测。
- 预测曲线是否跟随目标序列趋势。

对应 Notebook：`05_rnn_lstm_sequence_modeling.ipynb`。

## Attention 为什么改变了序列建模方式

RNN/LSTM 通常按顺序处理序列，远距离信息需要经过多个时间步传递。Attention 允许一个位置直接和其他位置建立关联，通过权重汇总相关信息。

Attention 的核心变化：

- 不只依赖单个隐状态保存历史。
- 可以直接计算任意两个位置之间的相关性。
- 权重矩阵提供了观察模型关注位置的方式。
- 更容易并行处理序列。

理解 Attention 时，可以先掌握 scaled dot-product attention：

```text
QK^T -> scale -> softmax -> weighted sum of V
```

对应 Notebook：`06_attention_transformer_basics.ipynb`。

## 推荐学习顺序

1. **Tensor 与 Autograd**：理解数据表示和自动求导。
2. **线性回归**：跑通第一个完整训练闭环。
3. **MLP 分类**：理解非线性建模和分类 loss。
4. **CNN 图像分类**：学习图像张量和局部特征表示。
5. **RNN/LSTM 序列建模**：学习序列样本构造和时间依赖。
6. **Attention/Transformer 基础**：理解注意力权重和序列关系建模。
7. **扩展实验**：尝试更多数据集、模型对比、训练日志和推理 Demo。

每一阶段都建议同时完成三件事：运行代码、观察结果、记录问题。这样比只阅读概念更容易形成稳定理解。
