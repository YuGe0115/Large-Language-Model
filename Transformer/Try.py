"""
Transformer模型完整实现
适合初学者学习的详细注释版本

主要组件：
1. 位置编码 (Positional Encoding)
2. 多头注意力机制 (Multi-Head Attention)
3. 前馈网络 (Feed Forward Network)
4. 编码器层 (Encoder Layer)
5. 解码器层 (Decoder Layer)
6. 完整Transformer模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# ==================== 配置参数 ====================
class Config:
    """模型配置类，方便调整参数"""
    def __init__(self):
        # 模型基本参数
        self.vocab_size = 10000      # 词汇表大小
        self.d_model = 512           # 模型维度
        self.n_heads = 8             # 注意力头数
        self.n_layers = 6            # 编码器/解码器层数
        self.d_ff = 2048             # 前馈网络隐藏层维度
        self.max_seq_len = 1000      # 最大序列长度
        self.dropout = 0.1           # Dropout概率
        
        # 训练参数
        self.batch_size = 32
        self.learning_rate = 0.0001

config = Config()

# ==================== 位置编码 ====================
class PositionalEncoding(nn.Module):
    """
    位置编码：为输入序列添加位置信息
    使用正弦和余弦函数生成位置编码
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 计算分母项：10000^(2i/d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        # 偶数位置使用sin，奇数位置使用cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 添加batch维度并注册为buffer（不参与梯度更新）
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        输入: x [seq_len, batch_size, d_model]
        输出: x + 位置编码
        """
        return x + self.pe[:x.size(0), :]

# ==================== 多头注意力机制 ====================
class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    将输入分成多个头，分别计算注意力，最后合并结果
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 每个头的维度
        
        # 线性变换层
        self.W_q = nn.Linear(d_model, d_model)  # Query变换
        self.W_k = nn.Linear(d_model, d_model)  # Key变换
        self.W_v = nn.Linear(d_model, d_model)  # Value变换
        self.W_o = nn.Linear(d_model, d_model)  # 输出变换
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        缩放点积注意力
        Q, K, V: [batch_size, n_heads, seq_len, d_k]
        """
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 应用mask（如果提供）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 应用softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 计算加权和
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 1. 线性变换并分割成多头
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 2. 计算注意力
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 3. 合并多头
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)
        
        # 4. 最终线性变换
        output = self.W_o(attention_output)
        
        return output, attention_weights

# ==================== 前馈网络 ====================
class FeedForward(nn.Module):
    """
    前馈网络：两层全连接网络
    第一层：d_model -> d_ff
    第二层：d_ff -> d_model
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        输入: x [batch_size, seq_len, d_model]
        输出: 经过前馈网络的结果
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

# ==================== 层归一化 ====================
class LayerNorm(nn.Module):
    """层归一化：对每个样本的特征进行归一化"""
    def __init__(self, d_model, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
        
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

# ==================== 编码器层 ====================
class EncoderLayer(nn.Module):
    """
    编码器层：包含多头注意力和前馈网络
    每个子层都有残差连接和层归一化
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # 自注意力 + 残差连接 + 层归一化
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

# ==================== 解码器层 ====================
class DecoderLayer(nn.Module):
    """
    解码器层：包含两个多头注意力层和一个前馈网络
    1. 掩码自注意力（防止看到未来信息）
    2. 编码器-解码器注意力
    3. 前馈网络
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # 1. 掩码自注意力
        attn_output, _ = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 2. 编码器-解码器注意力
        cross_attn_output, _ = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # 3. 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x

# ==================== 完整Transformer模型 ====================
class Transformer(nn.Module):
    """
    完整的Transformer模型
    包含编码器和解码器
    """
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.config = config
        
        # 词嵌入层
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(config.d_model, config.max_seq_len)
        
        # 编码器层
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])
        
        # 解码器层
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])
        
        # 输出层
        self.output_projection = nn.Linear(config.d_model, config.vocab_size)
        
        self.dropout = nn.Dropout(config.dropout)
        
    def create_padding_mask(self, seq):
        """创建填充掩码"""
        return (seq != 0).unsqueeze(1).unsqueeze(2)
    
    def create_look_ahead_mask(self, size):
        """创建前瞻掩码（防止解码器看到未来信息）"""
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        return mask == 0
    
    def encode(self, src, src_mask=None):
        """编码器前向传播"""
        # 词嵌入 + 位置编码
        x = self.embedding(src) * math.sqrt(self.config.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # 通过编码器层
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        
        return x
    
    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        """解码器前向传播"""
        # 词嵌入 + 位置编码
        x = self.embedding(tgt) * math.sqrt(self.config.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # 通过解码器层
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        
        return x
    
    def forward(self, src, tgt):
        """完整的前向传播"""
        # 创建掩码
        src_mask = self.create_padding_mask(src)
        tgt_mask = self.create_padding_mask(tgt)
        
        # 创建前瞻掩码
        tgt_look_ahead_mask = self.create_look_ahead_mask(tgt.size(1))
        tgt_mask = tgt_mask & tgt_look_ahead_mask.to(tgt.device)
        
        # 编码
        encoder_output = self.encode(src, src_mask)
        
        # 解码
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        
        # 输出投影
        output = self.output_projection(decoder_output)
        
        return output

# ==================== 训练和推理示例 ====================
def create_sample_data(config):
    """创建示例数据"""
    # 模拟输入序列
    src = torch.randint(1, config.vocab_size, (config.batch_size, 20))  # 源序列
    tgt = torch.randint(1, config.vocab_size, (config.batch_size, 15))  # 目标序列
    
    # 添加开始和结束标记
    src[:, 0] = 1  # 开始标记
    src[:, -1] = 2  # 结束标记
    tgt[:, 0] = 1
    tgt[:, -1] = 2
    
    return src, tgt

def train_example():
    """训练示例"""
    print("=== Transformer训练示例 ===")
    
    # 创建模型
    model = Transformer(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略填充标记
    
    # 创建示例数据
    src, tgt = create_sample_data(config)
    
    print(f"源序列形状: {src.shape}")
    print(f"目标序列形状: {tgt.shape}")
    
    # 前向传播
    model.train()
    output = model(src, tgt[:, :-1])  # 解码器输入不包含最后一个token
    
    print(f"模型输出形状: {output.shape}")
    
    # 计算损失
    loss = criterion(output.reshape(-1, config.vocab_size), tgt[:, 1:].reshape(-1))
    print(f"损失值: {loss.item():.4f}")
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print("训练步骤完成！")

def inference_example():
    """推理示例"""
    print("\n=== Transformer推理示例 ===")
    
    model = Transformer(config)
    model.eval()
    
    # 创建示例输入
    src = torch.randint(1, 100, (1, 10))  # 单个样本
    src[0, 0] = 1  # 开始标记
    src[0, -1] = 2  # 结束标记
    
    print(f"输入序列: {src}")
    
    # 简单的贪心解码
    with torch.no_grad():
        # 编码
        src_mask = model.create_padding_mask(src)
        encoder_output = model.encode(src, src_mask)
        
        # 逐步解码
        tgt = torch.tensor([[1]], dtype=torch.long)  # 开始标记
        
        for _ in range(20):  # 最大生成长度
            tgt_mask = model.create_padding_mask(tgt)
            tgt_look_ahead_mask = model.create_look_ahead_mask(tgt.size(1))
            tgt_mask = tgt_mask & tgt_look_ahead_mask.to(tgt.device)
            
            decoder_output = model.decode(tgt, encoder_output, src_mask, tgt_mask)
            output = model.output_projection(decoder_output)
            
            # 选择概率最高的词
            next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
            tgt = torch.cat([tgt, next_token], dim=1)
            
            # 如果生成了结束标记，停止
            if next_token.item() == 2:
                break
        
        print(f"生成序列: {tgt}")

# ==================== 主函数 ====================
if __name__ == "__main__":
    
    # 运行示例
    train_example()
    inference_example()
    
    print("框架搭建完成")