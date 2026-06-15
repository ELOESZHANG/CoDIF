

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Union, List, Dict, Any

from pcdet.models.roi_heads.Vtrans_layers import PatchEmbed
from pcdet.models.roi_heads.Vtrans_layers.block import Block
from pcdet.models.roi_heads.Vtrans_layers.rope import RotaryPositionEmbedding2D, PositionGetter
from pcdet.models.roi_heads.Vtrans_layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2

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
        # img_size=32, #518
        # in_chans=4,
        patch_size=(6,8), #14
        embed_dim=24, #1024
        depth=2, #24
        num_heads=4, #16
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        # patch_embed="dinov2_vits14_reg",
        aa_order=["local"],  #["frame", "global"]
        aa_block_size=2,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
    ):
        super().__init__()

        # self.__build_patch_embed__(patch_embed, img_size, in_chans, patch_size, num_register_tokens, embed_dim=embed_dim)

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

        # self.local2_blocks = nn.ModuleList(
        #     [
        #         block_fn(
        #             dim=embed_dim,
        #             num_heads=num_heads,
        #             mlp_ratio=mlp_ratio,
        #             qkv_bias=qkv_bias,
        #             proj_bias=proj_bias,
        #             ffn_bias=ffn_bias,
        #             init_values=init_values,
        #             qk_norm=qk_norm,
        #             rope=self.rope,
        #         )
        #         for _ in range(depth)
        #     ]
        # )

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))

        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = 1 #1 + num_register_tokens

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        # Register normalization constants as buffers
        # for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
        #     self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

        self.use_reentrant = False # hardcoded to False

    # def __build_patch_embed__(
    #     self,
    #     patch_embed,
    #     img_size,
    #     in_chans,
    #     patch_size,
    #     num_register_tokens,
    #     interpolate_antialias=True,
    #     interpolate_offset=0.0,
    #     block_chunks=0,
    #     init_values=1.0,
    #     embed_dim=1024,
    # ):
    #     """
    #     Build the patch embed layer. If 'conv', we use a
    #     simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
    #     """

        # if "conv" in patch_embed:
        #     self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        # else:
        #     vit_models = {
        #         "dinov2_vitl14_reg": vit_large,
        #         "dinov2_vitb14_reg": vit_base,
        #         "dinov2_vits14_reg": vit_small,
        #         "dinov2_vitg2_reg": vit_giant2,
        #     }
        #
        #     self.patch_embed = vit_models[patch_embed](
        #         img_size=img_size,
        #         in_chans=in_chans,
        #         patch_size=patch_size,
        #         num_register_tokens=num_register_tokens,
        #         interpolate_antialias=interpolate_antialias,
        #         interpolate_offset=interpolate_offset,
        #         block_chunks=block_chunks,
        #         init_values=init_values,
        #     )
        #
        #     # Disable gradient updates for mask token
        #     if hasattr(self.patch_embed, "mask_token"):
        #         self.patch_embed.mask_token.requires_grad_(False)

    def forward(self, patch_tokens: torch.Tensor) -> Tuple[List[torch.Tensor], int]:

        BS, H, W = patch_tokens.shape
        B=BS//32
        S=32

        # Expand camera and register tokens to match batch size and sequence length
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)

        # print('camera_token',camera_token.shape)
        # print('register_token', register_token.shape)
        # print('patch_tokens', patch_tokens.shape)

        # Concatenate special tokens with patch tokens
        tokens = torch.cat([camera_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size[0], W // self.patch_size[1], device=patch_tokens.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(patch_tokens.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        _, P, C = tokens.shape

        local_idx = 0
        local2_idx = 0
        output_list = []

        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "local":
                    tokens_0, local_idx, intermediates = self._process_local_attention(
                        tokens, B, S, P, C, local_idx, pos=pos
                    )
                    tokens = (tokens + tokens_0)*0.5
                elif attn_type == "local2":
                    tokens_0, local2_idx, intermediates = self._process_local2_attention(
                        tokens, B, S, P, C, local2_idx, pos=pos
                    )
                    tokens = (tokens + tokens_0)*0.5
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for i in range(len(intermediates)):

                concat_inter = intermediates[i]
                output_list.append(concat_inter)

        del concat_inter
        del intermediates

        return output_list

    def _process_local_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        """
        Process local attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(self.local_blocks[frame_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.local_blocks[frame_idx](tokens, pos=pos)
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, frame_idx, intermediates

    # def _process_local2_attention(self, tokens, B, S, P, C, global_idx, pos=None):
    #     """
    #     Process local2 attention blocks. We keep tokens in shape (B, S*P, C).
    #     """
    #     if tokens.shape != (B, S * P, C):
    #         tokens = tokens.view(B, S, P, C).view(B, S * P, C)
    #
    #     if pos is not None and pos.shape != (B, S * P, 2):
    #         pos = pos.view(B, S, P, 2).view(B, S * P, 2)
    #
    #     intermediates = []
    #
    #     # by default, self.aa_block_size=1, which processes one block at a time
    #     for _ in range(self.aa_block_size):
    #         if self.training:
    #             tokens = checkpoint(self.local2_blocks[global_idx], tokens, pos, use_reentrant=self.use_reentrant)
    #         else:
    #             tokens = self.local2_blocks[global_idx](tokens, pos=pos)
    #         global_idx += 1
    #         intermediates.append(tokens.view(B, S, P, C))
    #
    #     return tokens, global_idx, intermediates


def slice_expand_and_flatten(token_tensor, B, S):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """

    # Slice out the "query" tokens => shape (1, 1, ...)
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)

    # Finally flatten => shape (B*S, ...)
    combined = combined.view(B * S, *combined.shape[2:])
    return combined



if __name__ == '__main__':
    config = {
        'train_mode': 'finetune',
        'train_steps': 400,  # 500
        'sample_stepss': 10
    }
  ###128*128*216   128*16*8*6*36=128*16*48*36
    BS = 4096 #
    H = 36
    W = 48

    # Initialize the model and move to CUDA
    device = torch.device('cuda:0')

    model = Dino2Vtrans(patch_size=(6, 8), embed_dim=48).to(device)


    from torchinfo import summary

    print(summary(model))

    model.train()

    patch = torch.randn(BS, H, W).to(device)
    patch_tokens= model(patch)
    patch_fea = patch_tokens[1][:,:,5:,:]  #patch_tokens 2 layers;[0] 128,16,41,48;[1] 128,16,41,48

    print('patch_fea',patch_fea.shape)  #B,S,P,2C




