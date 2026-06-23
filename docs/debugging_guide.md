# PyTorch 常见错误排查指南

这份文档整理初学 PyTorch 时常见的训练错误。建议遇到问题时先缩小到最小可复现代码，再逐项检查 shape、dtype、loss、训练模式和数据划分。

## 1. Shape mismatch

**现象**  
报错包含 `size mismatch`、`mat1 and mat2 shapes cannot be multiplied`、`Expected input` 等信息。

**常见原因**

- 输入维度和模型层期望维度不一致。
- CNN flatten 后的特征数计算错误。
- batch 维、channel 维或 feature 维顺序写错。

**修正方式**

```python
print(x.shape)
x = x.view(x.size(0), -1)
print(x.shape)
```

**检查建议**

- Linear 层检查 `in_features`。
- CNN 检查输入是否是 `[N, C, H, W]`。
- RNN/LSTM 检查输入是否是 `[batch, seq_len, features]`。

## 2. Loss 不下降

**现象**  
训练多轮后 loss 基本不变，或者只在很小范围内波动。

**常见原因**

- 学习率不合适。
- 输入没有合适缩放。
- 标签格式错误。
- loss 函数使用方式不正确。
- 模型容量过小或数据本身模式不明显。

**修正方式**

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
x = (x - x.mean()) / (x.std() + 1e-8)
```

**检查建议**

先让模型在一个很小的数据子集上过拟合。如果小数据都学不会，优先检查训练循环、标签和 loss。

## 3. 梯度未清零

**现象**  
loss 变化异常，参数更新不稳定，训练结果难以复现。

**常见原因**  
PyTorch 默认累积梯度，每次 `backward()` 后梯度会叠加到已有 `.grad` 上。

**修正方式**

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

**检查建议**

把 `optimizer.zero_grad()` 放在每个 batch 反向传播之前。

## 4. 学习率设置不合理

**现象**

- 学习率过大：loss 震荡、发散或变成 `nan`。
- 学习率过小：loss 下降非常慢，看起来几乎不学习。

**常见原因**  
不同任务、模型和优化器需要不同的学习率范围。

**修正方式**

```python
# 常见起点
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 如果震荡明显，可以降低
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
```

**检查建议**

观察前 10 到 20 个 epoch 的 loss 曲线。如果大幅震荡，先降低学习率；如果几乎不变，可以适当提高。

## 5. CrossEntropyLoss 使用错误

**现象**  
分类训练效果差，或出现 dtype、shape 相关报错。

**常见原因**

- 在 `CrossEntropyLoss` 前手动使用 softmax。
- label 使用 one-hot，而不是类别索引。
- label dtype 不是 `torch.long`。
- 模型输出类别数和标签类别数不匹配。

**修正方式**

```python
logits = model(x)
labels = labels.long()
loss = torch.nn.CrossEntropyLoss()(logits, labels)
```

**检查建议**

- logits shape 应为 `[batch, num_classes]`。
- labels shape 通常为 `[batch]`。
- 不要先写 `torch.softmax(logits, dim=1)` 再传入 loss。

## 6. Train/Eval 模式错误

**现象**  
带 Dropout 或 BatchNorm 的模型训练和测试表现不稳定。

**常见原因**  
训练时没有调用 `model.train()`，评估时没有调用 `model.eval()`。

**修正方式**

```python
model.train()
# training loop

model.eval()
# evaluation loop
```

**检查建议**

在训练函数和评估函数开头明确设置模式，避免依赖模型当前状态。

## 7. `no_grad` 使用错误

**现象**

- 评估时内存占用偏高。
- 推理速度变慢。
- 在训练阶段误用 `no_grad` 导致无法反向传播。

**常见原因**  
评估时仍在构建计算图，或者训练时错误包裹了前向计算。

**修正方式**

```python
model.eval()
with torch.no_grad():
    logits = model(x_test)
```

**检查建议**

只在评估、推理或手动参数更新时使用 `torch.no_grad()`。训练前向和 loss 计算通常不应放在 `no_grad` 中。

## 8. 标签类型错误

**现象**  
loss 报 dtype 错误，或者训练结果异常。

**常见原因**

- 分类标签没有转成 `torch.long`。
- 回归目标误用了 long 类型。
- 标签 shape 和 loss 函数要求不匹配。

**修正方式**

```python
# 分类任务
labels = labels.long()

# 回归任务
targets = targets.float()
```

**检查建议**

分类任务通常使用类别索引和 `torch.long`；回归任务通常使用连续目标和 float。

## 9. 数据集划分错误

**现象**  
测试结果异常高或异常低，复现实验时波动很大。

**常见原因**

- 测试集样本混入训练集。
- 在划分前对全量数据做了带统计量的预处理。
- 分类任务划分后类别比例严重不均衡。

**修正方式**

```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)
```

**检查建议**

- 先划分训练集和测试集，再只用训练集拟合标准化器。
- 分类任务优先使用 `stratify=y` 保持类别比例。
- 固定 `random_state` 方便复现。

## 快速排查清单

- 输入 shape 是否符合模型要求。
- label dtype 是否符合 loss 要求。
- 每个 batch 是否调用 `optimizer.zero_grad()`。
- 分类任务是否把未归一化 logits 传给 `CrossEntropyLoss`。
- 训练和评估是否正确切换 `model.train()` / `model.eval()`。
- 评估和推理是否使用 `torch.no_grad()`。
- 学习率是否过大或过小。
- 数据划分是否独立、可复现。
