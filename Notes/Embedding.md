# Embedding 技术学习笔记

## 引言

Embedding（嵌入）是将离散的文本数据（如单词、字符或句子）映射到连续向量空间的技术，在自然语言处理（NLP）和大语言模型（LLM）中至关重要。Embedding 的目标是捕捉语义信息，使模型能够理解单词或短语之间的关系。常见的 Embedding 技术包括 One-hot、Word2Vec、FastText 和 GloVe。

本文将介绍这些技术的原理、优缺点，并结合 Python 代码示例展示实现。

## 1. One-hot Encoding

### 原理
One-hot Encoding 是一种简单的词嵌入方法，将每个单词表示为一个高维稀疏向量，向量长度等于词汇表大小，目标词的位置为 1，其余为 0。

- **优点**：
  - 简单直观，易于实现。
  - 无需训练，直接生成。
- **缺点**：
  - 向量维度随词汇表增大而膨胀，计算效率低。
  - 无法捕捉单词间的语义关系（如 "cat" 和 "dog" 的相似性）。
  - 稀疏性导致内存占用大。


## 2. Word2Vec

### 原理
Word2Vec 是 Google 在 2013 年提出的模型，通过神经网络学习低维稠密向量（通常 100-300 维）。它有两种架构：
- **CBOW（Continuous Bag of Words）**：用上下文预测目标词。
- **Skip-gram**：用目标词预测上下文。

Word2Vec 通过优化词向量，使语义相似的词在向量空间中更接近。

- **优点**：
  - 捕捉语义和语法关系（如 "king" - "man" + "woman" ≈ "queen"）。
  - 向量维度固定，计算效率高。
- **缺点**：
  - 无法处理未见词（OOV）。
  - 需要大量语料训练。

## 3. FastText

### 原理
FastText 是 Facebook 在 2016 年提出的改进版 Word2Vec。它将单词拆分为 n-gram 字符（如 "apple" 拆为 "ap", "pp", "ple"），并为每个 n-gram 学习向量。单词向量是其 n-gram 向量的和。

- **优点**：
  - 能处理未见词（OOV），因为基于字符 n-gram。
  - 对稀有词和形态变化（如 "running", "runner"）更鲁棒。
- **缺点**：
  - 模型复杂，训练时间较长。
  - 内存占用较高。

### 原理
GloVe（Global Vectors）是斯坦福大学在 2014 年提出的方法，基于全局词共现矩阵。它通过矩阵分解优化词向量，使向量间的点积近似于词共现概率的对数。

- **优点**：
  - 利用全局统计信息，捕捉语义关系。
  - 训练效率较高，适合大规模语料。
- **缺点**：
  - 无法处理未见词。
  - 需要预计算共现矩阵，内存需求较高。

# 假设已下载 GloVe 预训练向量（如 glove.6B.100d.txt）
# 转换为 Word2Vec 格式
```glove_input_file = "glove.6B.100d.txt"
word2vec_output_file = "glove.6B.100d.word2vec.txt"
glove2word2vec(glove_input_file, word2vec_output_file) ```

# 加载模型
```model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)```

# 测试
```print("Vector for 'cat':", model["cat"])
print("Similarity between 'cat' and 'dog':", model.similarity("cat", "dog"))
```

**说明**：

- 需要先下载 GloVe 预训练向量（例如从 GloVe 官网）。
- `glove2word2vec` 将 GloVe 格式转换为 Word2Vec 格式以便 `gensim` 使用。
- 实际运行需确保文件路径正确。

## 比较与应用

| 技术     | 维度       | 语义捕捉 | OOV 处理 | 训练复杂度 | 适用场景             |
| -------- | ---------- | -------- | -------- | ---------- | -------------------- |
| One-hot  | 词汇表大小 | 无       | 无       | 无需训练   | 简单任务，词汇表小   |
| Word2Vec | 100-300    | 强       | 无       | 中等       | 通用 NLP 任务        |
| FastText | 100-300    | 强       | 有       | 高         | 多语言、形态变化任务 |
| GloVe    | 50-300     | 强       | 无       | 中等       | 大规模语料，语义分析 |

### 在大模型中的应用
- **BERT**：使用 WordPiece 嵌入，结合上下文生成动态 Embedding。
- **GPT**：基于 BPE 的子词嵌入，结合 Transformer 架构。
- **优化策略**：预训练 Embedding + 微调、动态 Embedding（如 ELMo、BERT）。

## 总结

Embedding 技术是大模型的基础。One-hot 适合简单任务，但无法捕捉语义；Word2Vec 和 GloVe 通过语料训练生成稠密向量，适用于大多数 NLP 任务；FastText 进一步解决了 OOV 问题。实践时，推荐使用预训练 Embedding（如 Hugging Face 的 transformers 或 gensim），并根据任务需求选择合适的模型。
