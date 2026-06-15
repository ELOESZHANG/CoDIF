import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Union, List, Dict, Any

from pcdet.models.backbones_2d.fuser.Vtrans_layers import PatchEmbed
from pcdet.models.backbones_2d.fuser.Vtrans_layers.block import Block
from pcdet.models.backbones_2d.fuser.Vtrans_layers.rope import RotaryPositionEmbedding2D, PositionGetter
from pcdet.models.backbones_2d.fuser.Vtrans_layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2

# logger = logging.getLogger(__name__)

# _RESNET_MEAN = [0.485, 0.456, 0.406]
# _RESNET_STD = [0.229, 0.224, 0.225]


class Dino2Vtrans(nn.Module):
    """
    The Aggregator applies alternating-attention over input frames,
    as described in VGGT: Visual Geometry Grounded Transformer.

    Remember to set model.train() to enable gradient checkpointing to reduce memory usage.

    Args:
        img_size (int): Image size in pixels.
        patch_size (int): Size of each patch for PatchEmbed.
        embed_dim (int): Dimension of the token embeddings.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_register_tokens (int): Number of register tokens.
        block_fn (nn.Module): The block type used for attention (Block by default).
        qkv_bias (bool): Whether to include bias in QKV projections.
        proj_bias (bool): Whether to include bias in the output projection.
        ffn_bias (bool): Whether to include bias in MLP layers.
        patch_embed (str): Type of patch embed. e.g., "conv" or "dinov2_vitl14_reg".
        aa_order (list[str]): The order of alternating attention, e.g. ["frame", "global"].
        aa_block_size (int): How many blocks to group under each attention type before switching. If not necessary, set to 1.
        qk_norm (bool): Whether to apply QK normalization.
        rope_freq (int): Base frequency for rotary embedding. -1 to disable.
        init_values (float): Init scale for layer scale.
    """

    def __init__(
        self,
        in_channels=128,
        patch_size=(6,8),
        embed_dim=24,
        depth=1,
        num_heads=4,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        aa_order=["local"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
    ):
        super().__init__()
        patch_size = tuple(patch_size)  # 确保为 tuple，兼容 PyTorch Upsample

        # ----- 轻量 patch_embed: 1x1 conv 投影通道 → 平均池化聚合到 patch 网格 -----
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, 1, bias=False),
            nn.AvgPool2d(kernel_size=patch_size, stride=patch_size),
        )

        # ----- 输出投影: 上采样到原始分辨率 + 平滑回到 in_channels -----
        self.output_proj = nn.Sequential(
            nn.Upsample(scale_factor=patch_size, mode='bilinear', align_corners=False),
            nn.Conv2d(embed_dim, in_channels, 3, padding=1, bias=False),
        )

        # Initialize rotary position embedding if frequency > 0
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.local_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size

        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

        # 单 camera token（简化：不再需要 frame 维度）
        self.camera_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.patch_start_idx = 1

        nn.init.normal_(self.camera_token, std=1e-6)
        self.use_reentrant = False


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, H, W)  feature map
        Returns:
            (B, in_channels, H, W)    refined feature map
        """
        B, C, H, W = x.shape
        ph, pw = self.patch_size

        # Pad to patch-divisible size so patch_embed covers the full feature map
        Hpad = ((H + ph - 1) // ph) * ph
        Wpad = ((W + pw - 1) // pw) * pw
        if H != Hpad or W != Wpad:
            x = F.interpolate(x, (Hpad, Wpad), mode='bilinear', align_corners=False)

        # 1. patch_embed: (B, embed_dim, Hpad//ph, Wpad//pw)
        x = self.patch_embed(x)
        _, D, h, w = x.shape

        # 2. flatten patches → tokens: (B, num_patches, embed_dim)
        tokens = rearrange(x, 'b d h w -> b (h w) d')

        # 3. prepend camera token
        camera = self.camera_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        tokens = torch.cat([camera, tokens], dim=1)   # (B, 1+num_patches, embed_dim)

        # 4. position encoding（camera token 位置为 0）
        pos = None
        if self.rope is not None:
            pos = self.position_getter(B, h, w, device=x.device)
            pos_special = torch.zeros(B, 1, 2, device=x.device, dtype=pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # 5. transformer blocks
        for block in self.local_blocks:
            tokens = block(tokens, pos=pos)

        # 6. remove camera token
        tokens = tokens[:, 1:, :]  # (B, num_patches, embed_dim)

        # 7. reshape to spatial
        x = rearrange(tokens, 'b (h w) d -> b d h w', h=h, w=w)  # (B, embed_dim, h, w)

        # 8. output_proj → (B, in_channels, Hpad, Wpad)
        x = self.output_proj(x)

        # Crop back to original size if padded
        if H != Hpad or W != Wpad:
            x = x[:, :, :H, :W]

        return x


if __name__ == '__main__':
    config = {
        'train_mode': 'finetune',
        'train_steps': 400,
        'sample_stepss': 5
    }

    B = 2
    C = 128
    H = 176
    W = 176

    device = torch.device('cuda:0')

    model = Dino2Vtrans(in_channels=C, patch_size=(22, 8), embed_dim=176).to(device)

    from torchinfo import summary
    print(summary(model))

    model.train()

    # 输入: (B, in_channels, H, W)  feature map
    x = torch.randn(B, C, H, W).to(device)
    out = model(x)

    print('output', out.shape)  # (B, C, H, W)




