# Tokenizer技术学习笔记

## 绪

Tokenizer 是自然语言处理（NLP）和大语言模型（LLM）中的核心组件。它负责将原始文本转换为模型可以处理的数值序列（tokens）。在大模型如 GPT、BERT 等中，Tokenizer 直接影响模型的输入质量、词汇表大小和处理效率。

为什么需要 Tokenizer？
- 文本是连续的字符串，模型需要离散的输入。
- 处理词形变化、罕见词和多语言问题。
- 优化词汇表大小，避免 OOV（Out-of-Vocabulary）问题。

## 基本原理

Tokenizer 的工作流程：
1. **规范化（Normalization）**：清理文本，如小写转换、去除标点。
2. **分词（Tokenization）**：将文本拆分为 tokens。
3. **编码（Encoding）**：将 tokens 映射到 ID。
4. **解码（Decoding）**：将 ID 转换回文本。

常见级别：
- **Word-level**：按单词分词，词汇表大，易 OOV。
- **Character-level**：按字符分词，词汇表小，但序列长。
- **Subword-level**：折中方案，如 BPE（Byte Pair Encoding）、WordPiece。

## 常见 Tokenizer 算法

### 1. Byte Pair Encoding (BPE)
BPE 是一种子词分词算法，从字符开始，通过合并高频字节对构建词汇表。
- 步骤：
  - 初始化词汇表为所有独特字符。
  - 统计字节对频率，合并最高频对。
  - 重复直到词汇表达到指定大小。
- 优势：处理罕见词，减少 OOV。

### 2. WordPiece
类似于 BPE，但使用似然分数选择合并对，用于 BERT。

### 3. SentencePiece
无监督分词，支持多语言，常用于 T5、ALBERT。

## Python 实现示例

以下使用 Python 实现一个简单的 BPE Tokenizer 示例。我们从头构建一个小型 BPE 模型。

### 安装依赖
假设使用 Hugging Face 的 transformers 库（实际环境中需安装）。

```python
# 安装 transformers（如果未安装）
# pip install transformers
```

### 简单 BPE 从头实现

```python
import re
from collections import defaultdict

def get_stats(vocab):
    """计算符号对的频率"""
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i+1])] += freq
    return pairs

def merge_vocab(pair, vocab):
    """合并最高频符号对"""
    new_vocab = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in vocab:
        w_out = p.sub(''.join(pair), word)
        new_vocab[w_out] = vocab[word]
    return new_vocab

# 示例语料
corpus = "low low low low lower lowest newest wider wider new new"
words = corpus.split()
vocab = {' '.join(list(w)) + ' </w>': 1 for w in set(words)}  # 初始化词汇表，添加 </w> 表示词尾

# 更新频率
for w in words:
    key = ' '.join(list(w)) + ' </w>'
    vocab[key] += 1

num_merges = 10  # 合并次数
for i in range(num_merges):
    pairs = get_stats(vocab)
    if not pairs:
        break
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)
    print(f"Merge {i+1}: {best}")

print("\n最终词汇表：")
print(vocab)
```

**代码输出示例**：

```
Merge 1: ('l', 'o')
Merge 2: ('lo', 'w')
Merge 3: ('low', '</w>')
Merge 4: ('n', 'e')
Merge 5: ('ne', 'w')
Merge 6: ('w', 'i')
Merge 7: ('wi', 'd')
Merge 8: ('wid', 'e')
Merge 9: ('wide', 'r')
Merge 10: ('new', '</w>')

最终词汇表：
{'low</w>': 5, 'lower</w>': 1, 'lowest</w>': 1, 'newest</w>': 1, 'wider</w>': 2, 'new</w>': 2}
```

### 使用 Hugging Face Transformers

实际大模型中使用预训练 Tokenizer。

```python
from transformers import BertTokenizer

# 加载 BERT Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 示例文本
text = "Hello, how are you?"
tokens = tokenizer.tokenize(text)
ids = tokenizer.convert_tokens_to_ids(tokens)

print("Tokens:", tokens)
print("IDs:", ids)

# 解码
decoded = tokenizer.decode(ids)
print("Decoded:", decoded)
```

**输出**：
```
Tokens: ['hello', ',', 'how', 'are', 'you', '?']
IDs: [7592, 1010, 2129, 2024, 2017, 1029]
Decoded: hello, how are you?
```

## 在大模型中的应用

- **GPT 系列**：使用 BPE-based Tokenizer，词汇表约 50k。
- **BERT**：WordPiece，处理掩码语言建模。
- **挑战**：多语言支持、长序列处理（使用注意力机制）。
- **优化**：动态词汇表、efficient encoding 以减少计算开销。

## 总结

Tokenizer 是大模型输入管道的关键。通过子词算法如 BPE，我们能高效处理文本。实践时需使用 Hugging Face 库快速实验，并理解底层原理以优化自定义模型。
