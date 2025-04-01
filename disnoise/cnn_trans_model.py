import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义简单的CNN模块

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


# 定义简单的Transformer模块
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop)  # 修改这里
        self.dropout = nn.Dropout(drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.ReLU(), # 使用ReLU激活函数替代GELU，因为GELU在某些PyTorch版本中可能不可用
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x1 = self.norm1(x)
        attn_output, _ = self.attn(x1, x1, x1)
        x = x + self.dropout(attn_output)
        x2 = self.norm2(x)
        x = x + self.mlp(x2)
        return x


# 结合CNN和Transformer的模型，用于图像去噪
class HybridModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_cnn_blocks=3, num_transformer_blocks=2, dim=256):
        super(HybridModel, self).__init__()
        self.cnn_blocks = nn.ModuleList([
            CNNBlock(in_channels if i == 0 else dim, dim) for i in range(num_cnn_blocks)
        ])
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(dim) for _ in range(num_transformer_blocks)
        ])
        self.final_conv = nn.Conv2d(dim, out_channels, kernel_size=3, padding=1)
        # 添加跳跃连接，保留输入信息
        self.skip_connection = True

    def forward(self, x):
        # 保存输入用于跳跃连接
        input_x = x
        
        # CNN部分
        for cnn_block in self.cnn_blocks:
            x = cnn_block(x)
        # 将特征图转换为序列
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        # Transformer部分
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        # 将序列转换回特征图
        x = x.transpose(1, 2).view(b, c, h, w)
        # 最终卷积层
        x = self.final_conv(x)
        
        # 添加跳跃连接，模型学习残差（噪声）
        if self.skip_connection:
            x = input_x - x  # 预测噪声并从输入中减去
        
        return x

# 测试代码
if __name__ == "__main__":
    model = HybridModel()
    # 创建一个干净的图像
    clean_image = torch.randn(1, 3, 256, 256)
    # 添加噪声
    noise = torch.randn(1, 3, 256, 256) * 0.1
    noisy_image = clean_image + noise
    
    # 将噪声图像输入模型
    denoised_image = model(noisy_image)
    
    print(f"输入噪声图像形状: {noisy_image.shape}")
    print(f"输出去噪图像形状: {denoised_image.shape}")
    
    # 计算去噪效果
    noise_level = torch.mean((noisy_image - clean_image) ** 2)
    denoised_level = torch.mean((denoised_image - clean_image) ** 2)
    print(f"原始噪声水平: {noise_level.item():.6f}")
    print(f"去噪后噪声水平: {denoised_level.item():.6f}")

    