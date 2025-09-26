# XGBoost

## 1. XGBoost 简介

XGBoost（eXtreme Gradient Boosting）是一种高效的梯度提升算法，广泛应用于机器学习任务，如分类、回归和排序问题。它基于梯度提升决策树（GBDT），通过优化损失函数和正则化来提升模型性能。

### 特点
- **高性能**：支持并行计算，速度快。
- **灵活性**：支持自定义损失函数，适用于多种任务。
- **正则化**：内置 L1 和 L2 正则化，防止过拟合。
- **缺失值处理**：自动处理缺失值。

## 2. XGBoost 基本用法

以下是一个简单的 XGBoost 分类示例，使用 Python 的 `xgboost` 库。

### 示例代码：分类任务

```python
import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 加载数据集
iris = load_iris()
X, y = iris.data, iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建 XGBoost 模型
model = xgb.XGBClassifier(objective='multi:softmax', num_class=3, max_depth=3, learning_rate=0.1, n_estimators=100)

# 训练模型
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确率: {accuracy:.2f}")
```

**代码说明**：
- `objective='multi:softmax'`：多分类任务，使用 softmax 损失函数。
- `num_class=3`：指定类别数量（鸢尾花数据集有 3 类）。
- `max_depth=3`：树的最大深度，控制模型复杂度。
- `learning_rate=0.1`：学习率，控制每次迭代的步长。
- `n_estimators=100`：提升树的棵数。

## 3. 重要参数调优

- **学习率（learning_rate）**：通常设置为 0.01-0.3。值越小，模型越稳健，但需要更多棵树。
- **最大深度（max_depth）**：控制树的深度，通常设置为 3-10。
- **子样本比例（subsample）**：每次迭代使用的样本比例，防止过拟合，通常为 0.5-1.0。
- **正则化参数**：
  - `lambda`：L2 正则化权重。
  - `alpha`：L1 正则化权重。

### 示例代码：参数调优

```python
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

# 定义参数网格
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [50, 100, 200]
}

# 创建 XGBoost 模型
model = xgb.XGBClassifier(objective='multi:softmax', num_class=3)

# 网格搜索
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# 输出最佳参数
print("最佳参数:", grid_search.best_params_)
print("最佳得分:", grid_search.best_score_)
```

**代码说明**：
- 使用 `GridSearchCV` 进行网格搜索，自动寻找最佳参数组合。
- `cv=5`：5 折交叉验证。
- `scoring='accuracy'`：以准确率为评估指标。

## 4. 特征重要性

XGBoost 可以输出特征的重要性，帮助理解模型的决策依据。

### 示例代码：特征重要性可视化

```python
import matplotlib.pyplot as plt

# 绘制特征重要性
xgb.plot_importance(model)
plt.show()
```

**代码说明**：
- `xgb.plot_importance`：绘制特征重要性图，显示每个特征对模型预测的贡献。

## 5. 优缺点

### 优点
- 高精度：通过梯度提升和正则化，模型性能优于许多其他算法。
- 高效：支持并行计算和 GPU 加速。
- 灵活：支持多种任务和自定义损失函数。

### 缺点
- 参数较多，调优复杂。
- 对噪声数据敏感，需仔细预处理数据。
- 训练时间可能较长，尤其是数据量大时。

## 6. 总结

XGBoost 是一个强大的机器学习工具，适合处理结构化数据。通过合理调参和数据预处理，可以显著提升模型性能。