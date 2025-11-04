import torch
import torch.nn as nn
import torch.nn.functional as F

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU is available! Using device: {device}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print(f"GPU not available. Using device: {device}")

class SimpleAttention(nn.Module):
    """
    一个简单的自注意力模块。
    它包含三个线性层（用于生成 Query, Key, Value）和最终的输出投影。
    """
    def __init__(self, embed_dim):
        """
        embed_dim: 输入嵌入的维度
        """
        super().__init__()
        self.embed_dim = embed_dim

        # 定义 Query, Key, Value 的线性变换
        # 这三个是可学习的参数矩阵
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        # 定义输出的投影层（可选，但在 Transformer 中很常见）
        self.proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        """
        x: 输入张量,  shape: (batch_size, seq_len, embed_dim)
        """
        # 1. 生成 Q, K, V
        Q = self.query(x)  # (batch_size, seq_len, embed_dim)
        K = self.key(x)    # (batch_size, seq_len, embed_dim)
        V = self.value(x)  # (batch_size, seq_len, embed_dim)

        # 2. 计算 Query 和 Key 的点积
        # 为了进行矩阵乘法，我们交换 K 的最后两个维度
        # Q.shape: (batch_size, seq_len, embed_dim)
        # K.permute(0, 2, 1).shape: (batch_size, embed_dim, seq_len)
        # scores.shape: (batch_size, seq_len, seq_len)
        scores = torch.matmul(Q, K.permute(0, 2, 1))
        
        # 3. 缩放点积，防止梯度过小
        # embed_dim 的平方根是标准的缩放因子
        scores = scores / (self.embed_dim ** 0.5)
        
        # 4. 通过 Softmax 获得注意力权重
        # 在最后一个维度上计算 softmax，使得每一行的和为 1
        attention_weights = F.softmax(scores, dim=-1)
        
        # 5. 将注意力权重应用到 Value 上
        # attention_weights.shape: (batch_size, seq_len, seq_len)
        # V.shape: (batch_size, seq_len, embed_dim)
        # output.shape: (batch_size, seq_len, embed_dim)
        output = torch.matmul(attention_weights, V)
        
        # 6. 最终的线性投影
        output = self.proj(output)
        
        return output, attention_weights


if __name__ == "__main__":
    batch_size = 4  
    seq_len = 10    
    embed_dim = 256  

    model = SimpleAttention(embed_dim=embed_dim)
    model.to(device)  

    print("\nModel Architecture:")
    print(model)
    input_tensor = torch.randn(batch_size, seq_len, embed_dim).to(device)

    model.eval() 

    from rocprofsys.profiler import config
    from rocprofsys import profile
    config.include_args = True
    config.include_filename = True
    config.include_line = False

    with torch.no_grad():  
        with profile():
            output, weights = model(input_tensor)

    print("\nInput tensor shape:", input_tensor.shape)
    print("Output tensor shape:", output.shape)
    print("Attention weights shape:", weights.shape)

    print("\n--- Sample Output ---")
    print("Output for the first sequence, first token:")
    print(output[0, 0, :5])
    
    print("\nAttention weights for the first sequence (how it attends to all tokens):")
    print(weights[0])
