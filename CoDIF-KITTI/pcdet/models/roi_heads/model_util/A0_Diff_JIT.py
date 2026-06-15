import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from einops import rearrange
import math


class PositionEmbedding(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, embed_dim))

    def forward(self, x):
        # print(x.shape)
        # print(self.pos_embedding.shape)
        return x + self.pos_embedding


class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None, key_padding_mask=None):
        # 自注意力层
        attn_output, _ = self.attn(x, x, x, attn_mask=attention_mask, key_padding_mask=key_padding_mask)

        # 残差连接
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # MLP层
        mlp_output = self.mlp(x)
        x = x + self.dropout(mlp_output)
        x = self.norm2(x)

        return x

def timestep_embedding(t, dim, max_period=10000):
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=dim // 2, dtype=torch.float32) / (dim // 2)).to(
        device=t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return embedding

class JIT(nn.Module):
    def __init__(self, patch_size=16, embed_dim=768, num_heads=12, num_layers=12, hidden_dim=3072):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # 将图像分块
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

        # 位置嵌入
        self.pos_embed = PositionEmbedding((256 // patch_size) ** 2, embed_dim)

        # Transformer解码器块
        self.blocks = nn.Sequential(*[
            TransformerDecoderBlock(embed_dim, num_heads, hidden_dim)
            for _ in range(num_layers)
        ])

        # 最终输出层（将序列转换回图像）
        self.output_layer = nn.Linear(embed_dim, 3 * patch_size ** 2)

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, x, t):
        # 输入形状: [batch_size, 3, H, W]
        batch_size = x.shape[0]

        # 将图像分块并嵌入
        patches = self.patch_embed(x)  # [batch_size, embed_dim, grid, grid]
        # print('patches',patches.shape)
        patches = rearrange(patches, 'b c h w -> b (h w) c')

        # 添加位置嵌入
        embedded = self.pos_embed(patches)

        # 通过Transformer块
        for block in self.blocks:
            embedded = block(embedded)

        # 输出层
        output = self.output_layer(embedded)

        # 将序列转换回图像块
        grid_size = x.shape[2] // self.patch_size
        output = rearrange(output, 'b (h w) (c p1 p2) -> b c (h p1) (w p2)',
                           h=grid_size, w=grid_size, p1=self.patch_size, p2=self.patch_size)

        return output


def train_step(model, x, optimizer):
    model.train()

    t = torch.rand(x.shape[0], device=x.device)  # sample t
    e = torch.randn_like(x)  # noise
    z = t.unsqueeze(-1).unsqueeze(-1) * x + (1 - t.unsqueeze(-1).unsqueeze(-1)) * e
    v = (x - z) / (1 - t.unsqueeze(-1).unsqueeze(-1))

    x_pred = model(z, t)
    v_pred = (x_pred - z) / (1 - t.unsqueeze(-1).unsqueeze(-1))

    loss = torch.mean((v - v_pred) ** 2)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


def sample(model, batch_size, img_size, patch_size):
    model.eval()
    with torch.no_grad():
        z = torch.randn(batch_size, 3, img_size, img_size)
        t = 1.0

        while t > 0.0:
            t_next = max(t - 0.05, 0.0)
            x_pred = model(z, torch.full((batch_size,), t, device=z.device))
            v_pred = (x_pred - z) / (1 - t)
            z = z + (t_next - t) * v_pred
            t = t_next
    return z


def main():
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)

    # 创建模型实例
    model = JIT(patch_size=16, embed_dim=256, num_heads=4, num_layers=2, hidden_dim=512)

    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 创建单个样本数据 (1张32x32的小图像)
    img_size = 256
    x = torch.randn(1, 3, img_size, img_size)  # 形状: [batch_size, channels, height, width]

    print("开始训练步骤...")
    # 执行一次训练步骤
    loss = train_step(model, x, optimizer)
    print(f"训练损失: {loss.item():.4f}")

    print("开始采样...")
    # 生成样本图像
    sample_img = sample(model, batch_size=1, img_size=img_size, patch_size=16)
    print(f"采样图像形状: {sample_img.shape}")  # 应为 [1, 3, 32, 32]

    # 检查输出是否合理
    assert sample_img.shape == (1, 3, img_size, img_size), "采样图像形状不正确"

    print("测试成功完成！")


if __name__ == "__main__":
    main()
