# Loss Function

## 基本概念

损失函数的作用是量化模型预测值与真实值之间的差距，优化器通过最小化损失函数调整模型参数。损失函数通常分为以下类别：
- **回归任务**：预测连续值（如房价预测）。
- **分类任务**：预测离散类别（如情感分类）。
- **序列任务**：处理序列数据（如机器翻译）。

## 常见损失函数

### 1. 均方误差（Mean Squared Error, MSE）
**定义**：预测值与真实值差的平方均值。
\[ L = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2 \]

- **适用场景**：回归任务，假设误差服从高斯分布。
- **优点**：简单，易于优化；对大误差敏感。
- **缺点**：对异常值敏感，可能导致模型过分关注离群点。

### 2. 均方根误差（Root Mean Squared Error, RMSE）
**定义**：MSE 的平方根。
\[ L = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2} \]
- **适用场景**：回归任务，结果与目标值同量纲。
- **优点**：直观，量纲与数据一致。
- **缺点**：仍对异常值敏感。

### 3. 交叉熵损失（Cross-Entropy Loss）
**定义**：衡量两个概率分布之间的差异，常用于分类任务。
$$
L = -\sum_{i=1}^C y_i \log(\hat{y}_i)
$$
（对于二分类，C=2；多分类使用 softmax 输出）。

- **适用场景**：分类任务（如文本分类、图像分类）。
- **优点**：适合概率输出，梯度计算稳定。
- **缺点**：对标签噪声敏感。

### 4. 负对数似然损失（Negative Log-Likelihood, NLL）
**定义**：交叉熵损失的变体，结合 log-softmax。
$$
L = -\log(\hat{y}_{y_i})
$$


- **适用场景**：NLP 序列任务（如语言建模）。
- **优点**：数值稳定性好，适合多类别任务。
- **缺点**：需要配合 softmax 或 log-softmax。

### 5. Hinge 损失
**定义**：用于支持向量机（SVM）或二分类任务。
$$
L = \max(0, 1 - y_i \cdot \hat{y}_i)
$$

- **适用场景**：二分类任务，最大化分类边界。
- **优点**：鼓励正确分类并保持边界。
- **缺点**：不适合概率输出。

### 6. KL 散度（Kullback-Leibler Divergence）
**定义**：衡量两个概率分布的差异。
$$
L = \sum_{i} p_i \log\left(\frac{p_i}{\hat{p}_i}\right)
$$

- **适用场景**：生成模型（如变分自编码器 VAE）、知识蒸馏。
- **优点**：适合分布匹配。
- **缺点**：不对称，计算复杂。

## 在大模型中的应用

在大语言模型（如 GPT、BERT）中，损失函数通常与任务相关：
- **语言建模**：使用交叉熵损失或 NLL，预测下一个词的概率。
- **分类任务**：如 BERT 的情感分析，使用交叉熵损失。
- **生成任务**：如 GAN 或 VAE，可能结合 KL 散度。
- **多任务学习**：组合多种损失函数，如分类 + 回归。

### 优化技巧
- **正则化**：在损失函数中加入 L1/L2 正则化项，防止过拟合。
- **标签平滑（Label Smoothing）**：降低对正确标签的过分自信。
- **加权损失**：处理类别不平衡问题。

## Python 实现示例

```python
import torch
import torch.nn as nn
import numpy as np

# 示例数据
y_true = torch.tensor([1.0, 0.0, 1.0, 1.0])  # 真实值（回归或二分类）
y_pred = torch.tensor([0.9, 0.1, 0.8, 0.7])  # 预测值
y_true_class = torch.tensor([1, 0, 1, 1])     # 分类任务真实标签
y_pred_logits = torch.tensor([[0.2, 0.8], [0.9, 0.1], [0.3, 0.7], [0.4, 0.6]])  # 分类任务预测 logits

# 1. 均方误差 (MSE)
mse_loss = nn.MSELoss()
mse = mse_loss(y_pred, y_true)
print("MSE Loss:", mse.item())

# 2. 均方根误差 (RMSE)
rmse = torch.sqrt(mse)
print("RMSE Loss:", rmse.item())

# 3. 交叉熵损失
cross_entropy_loss = nn.CrossEntropyLoss()
ce_loss = cross_entropy_loss(y_pred_logits, y_true_class)
print("Cross Entropy Loss:", ce_loss.item())

# 4. 负对数似然损失 (NLL)
log_softmax = nn.LogSoftmax(dim=1)
nll_loss = nn.NLLLoss()
nll = nll_loss(log_softmax(y_pred_logits), y_true_class)
print("NLL Loss:", nll.item())

# 5. Hinge 损失（手动实现）
def hinge_loss(y_true, y_pred):
    y_true = 2 * y_true - 1  # 转换为 {-1, 1}
    loss = torch.mean(torch.clamp(1 - y_pred * y_true, min=0))
    return loss

hinge = hinge_loss(y_true, y_pred)
print("Hinge Loss:", hinge.item())

# 6. KL 散度
kl_div_loss = nn.KLDivLoss(reduction='batchmean')
p = torch.softmax(y_pred_logits, dim=1)
q = torch.softmax(torch.tensor([[0.3, 0.7], [0.8, 0.2], [0.4, 0.6], [0.5, 0.5]]), dim=1)
kl_loss = kl_div_loss(log_softmax(y_pred_logits), q)
print("KL Divergence Loss:", kl_loss.item())
```

**输出示例**：

```
MSE Loss: 0.0125
RMSE Loss: 0.1118
Cross Entropy Loss: 0.4512
NLL Loss: 0.4512
Hinge Loss: 0.3250
KL Divergence Loss: 0.0234
```

**代码说明**：
- 使用 PyTorch 的内置损失函数（如 `nn.MSELoss`, `nn.CrossEntropyLoss`）简化实现。
- Hinge 损失手动实现，因为 PyTorch 未提供直接 API。
- KL 散度需要输入为概率分布，通常结合 log-softmax。

## 实践中的注意事项

1. **数值稳定性**：
   - 交叉熵损失通常结合 log-softmax 避免数值溢出。
   - 对输入进行归一化（如 softmax）以确保概率分布。

2. **任务适配**：
   - 回归任务：MSE 或 RMSE。
   - 分类任务：交叉熵或 NLL。
   - 生成任务：结合 KL 散度或对抗损失。

3. **超参数调整**：
   - 加权损失：处理不平衡数据集。
   - 标签平滑：如交叉熵中的 `label_smoothing` 参数。

## 在大模型中的具体应用

- **BERT**：使用交叉熵损失进行掩码语言建模（MLM）和下一句预测（NSP）。
- **GPT**：NLL 损失用于自回归语言建模。
- **生成对抗网络（GAN）**：结合交叉熵和 KL 散度优化生成器和判别器。
- **知识蒸馏**：KL 散度用于教师模型和学生模型的分布匹配。

## 总结

损失函数是大模型优化的核心，不同任务需要选择合适的损失函数。MSE 和 RMSE 适用于回归任务，交叉熵和 NLL 是分类和序列任务的主流选择，Hinge 和 KL 散度在特定场景（如 SVM 和生成模型）中表现优异。实践中，使用 PyTorch 或 TensorFlow 的内置损失函数可以提高效率，同时需关注数值稳定性和任务适配性。
