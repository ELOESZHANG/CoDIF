import torch
import torch.nn as nn
import torch.nn.functional as F
from pcdet.models.backbones_3d.pointnet2_backbone import PointNet2Backbone, PointNet2MSG
from torch.nn.functional import normalize

class ColorEh(nn.Module):
    def color_fc(self, in_channel=9, out_channels=32):
        self.fc1 = nn.Linear(in_channel, out_channels)
        self.fc2 = nn.Linear(out_channels, out_channels)
        self.fc3 = nn.Linear(out_channels, out_channels)
        # self.dp1 = nn.Dropout(p=0.05)
        # self.dp2 = nn.Dropout(p=0.05)
        self.relu1 = nn.ReLU()
        # self.relu2 = nn.ReLU()

        FC = nn.Sequential(
            self.fc1,
            self.fc2,
            self.fc3,
            self.relu1
        )
        return FC

    def __init__(self):
        super(ColorEh, self).__init__()
        self.color_fc11 = self.color_fc(6, 18)
        self.color_fc21 = self.color_fc(18, 54)
        self.color_fc31 = self.color_fc(54, 18)
        self.color_fc41 = self.color_fc(18, 6)

        self.color_fc22 = self.color_fc(6, 54)
        self.color_fc23 = self.color_fc(486, 54)



    def forward(self, color_point_fea, color_point_link):
        if color_point_fea.shape[0] == 0:
            return color_point_fea
        # color_point_fea [ **,9]
        # color_point_link [ **,90]

        N, M = color_point_link.shape
        point_empty = (color_point_link == 0).nonzero()  # select no zero
        color_point_link[point_empty[:, 0], point_empty[:, 1]] = point_empty[:, 0]
        color_point_link = color_point_link.view(-1)

        ninei = torch.index_select(color_point_fea, 0, color_point_link)
        ninei = ninei.view(N, M, -1)
        nine0 = color_point_fea.unsqueeze(dim=-2).repeat([1, M, 1])
        ninei = ninei - nine0

        color_point_fea[:, 3:6] /= 255.0
        color_point_fea[:, :3] = normalize(color_point_fea[:, :3], dim=0)
        color_point_fea[:, 6:] = normalize(color_point_fea[:, 6:], dim=0)

        ninei = ninei[:, :, [0, 1, 2, 6, 7, 8]]

        fea1 = self.color_fc11(color_point_fea[:, :6])
        fea2 = self.color_fc21(fea1)
        fea3 = self.color_fc31(fea2)
        fea4 = self.color_fc41(fea3)

        fea2_1 = torch.index_select(fea2, 0, color_point_link).view(N, M, -1)
        fea2_1 = fea2_1 * self.color_fc22(ninei)
        fea2_1 = self.color_fc23(fea2_1.view(N, -1))

        color_conv_fea = torch.cat([fea4, fea3, fea2_1, fea1, color_point_fea[:, :6]], dim=-1) #[50001,102]

        return color_conv_fea

class TransAttention(nn.Module):
    def __init__(self, channels):
        super(TransAttention, self).__init__()
        self.channels = channels

        self.fc1 = nn.Sequential(nn.Linear(channels, channels),
                                 nn.Linear(channels, channels),
                                 nn.Linear(channels, channels),
                                 nn.Linear(channels, channels),
                                 nn.SELU(),
                                 nn.Dropout(p=0.1, inplace=False),
                                 nn.Linear(channels, channels),
                                 nn.Linear(channels, channels),
                                 nn.Linear(channels, channels),
                                 )


    def forward(self, pseudo_feas0, valid_feas0):
        B,N,_ = pseudo_feas0.size()
        dn = N
        Ra=1
        aaa=0.1 #0.1

        # pseudo_feas0 = normalize(pseudo_feas0, dim=-1)
        # valid_feas0  = normalize(valid_feas0, dim=-1)
        pseudo_feas = pseudo_feas0.transpose(1, 2)
        valid_feas = valid_feas0.transpose(1, 2)

        pse_Q = self.fc1(pseudo_feas)
        pse_K = self.fc1(pseudo_feas)
        pse_V = pseudo_feas
        pse_Q = F.softmax(pse_Q, dim=-2)
        pse_K = F.softmax(pse_K, dim=-1)

        val_Q = self.fc1(valid_feas)
        val_K = self.fc1(valid_feas)
        val_V = valid_feas
        val_Q = F.softmax(val_Q, dim=-2)
        val_K = F.softmax(val_K, dim=-1)

        pseudo_feas_end = torch.bmm(pse_Q, val_K.transpose(-2, -1)) / dn
        # pseudo_feas_end = F.relu(pseudo_feas_end)
        pseudo_feas_end = torch.bmm(pseudo_feas_end, pse_V)
        pseudo_feas_end = self.fc1(pseudo_feas_end).transpose(1, 2)
        pseudo_feas_end = normalize(pseudo_feas_end, dim=-1)*aaa + pseudo_feas0*(1.1-0.1*Ra)

        valid_feas_end = torch.bmm(val_Q, pse_K.transpose(-2, -1)) / dn
        # valid_feas_end = F.relu(valid_feas_end)
        valid_feas_end = torch.bmm(valid_feas_end, val_V)
        valid_feas_end = self.fc1(valid_feas_end).transpose(1, 2)
        valid_feas_end = normalize(valid_feas_end, dim=-1)*aaa + valid_feas0*(1.1-0.1*Ra)
        # print('pseudo_features_att', pseudo_features_att.shape)

        return pseudo_feas_end, valid_feas_end


class ROIAttention(nn.Module):
    def __init__(self, channels):
        super(ROIAttention, self).__init__()
        self.channels = channels

        self.fc1 = nn.Linear(self.channels * 2, self.channels * 4)
        self.fc2 = nn.Linear(self.channels * 4, self.channels * 2)
        self.fc3 = nn.Linear(self.channels * 2, self.channels)

        self.fc4p = nn.Linear(self.channels//2, self.channels//4)
        self.fc4v = nn.Linear(self.channels//2, self.channels//4)
        self.fc5p = nn.Linear(self.channels//4, 1)
        self.fc5v = nn.Linear(self.channels//4, 1)

        self.conv1 = nn.Sequential(nn.Conv1d(self.channels, self.channels, 1),
                                    nn.BatchNorm1d(self.channels),
                                    nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(self.channels, self.channels, 1),
                                    nn.BatchNorm1d(self.channels),
                                    nn.ReLU())

    def forward(self, pse_feas, val_feas):
        # print('pseudo_feas',pseudo_feas.shape)       #[100, 128, 216])
        Rb=1
        B, N, _  = pse_feas.size()
        pse_feas_1 = pse_feas.transpose(1,2).reshape(-1,N)       #[100,216,128]
        val_feas_1 = val_feas.transpose(1,2).reshape(-1,N)

        fusion_fea = torch.cat([pse_feas_1, val_feas_1], dim=-1)  #[100,216,256]
        fusion_fea = self.fc1(fusion_fea)
        fusion_fea = self.fc2(fusion_fea)
        fusion_fea = self.fc3(fusion_fea)
        C = self.channels//2
        pse_feas_1 = fusion_fea[:, :C]
        val_feas_1 = fusion_fea[:, C:]

        pse_feas_1 = self.fc4p(pse_feas_1)
        val_feas_1 = self.fc4v(val_feas_1)
        pse_feas_1 = self.fc5p(pse_feas_1)
        val_feas_1 = self.fc5v(val_feas_1)

        pse_feas_1 = torch.sigmoid(pse_feas_1).view(B, -1, 1).transpose(1, 2)
        val_feas_1 = torch.sigmoid(val_feas_1).view(B, -1, 1).transpose(1, 2)

        pse_feas_end = self.conv1(pse_feas * pse_feas_1*(1.1-0.1*Rb))  # [100,1,216]
        val_feas_end = self.conv2(val_feas * val_feas_1*(1.1-0.1*Rb))

        return pse_feas_end, val_feas_end


class Baseline_color(nn.Module):
    def __init__(self):
        super(Baseline_color, self).__init__()

    def forward(self, points_features, points_neighbor):
        if points_features.shape[0] == 0:
            return points_features
        #points_features [ **,9]
        #points_neighbor [ **,9]

        points_features[:, 3:6] /= 255.0
        points_features[:, :3] = normalize(points_features[:, :3], dim=0)
        points_features[:, 6:] = normalize(points_features[:, 6:], dim=0)

        N, _ = points_neighbor.shape
        point_empty = (points_neighbor == 0).nonzero()  #select no zero
        points_neighbor[point_empty[:, 0], point_empty[:, 1]] = point_empty[:, 0]
        points_neighbor=points_neighbor.view(-1)

        xyz_aaa = torch.index_select(points_features, 0, points_neighbor).view(N,-1)

        pointnet_feas = torch.cat([xyz_aaa, points_features], dim=-1)
        # points_features [ **,90]
        return pointnet_feas


#######DWconv
class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   stride, padding, dilation,
                                   groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class ChannelAttention(nn.Module):
    """通道注意力机制"""

    def __init__(self, in_channels, reduction_ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels * reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class DSConvWithChannelAttention(nn.Module):
    """深度可分离卷积与通道注意力交互模块"""

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1,
                 reduction_ratio=4):
        super(DSConvWithChannelAttention, self).__init__()

        self.ds_conv = DepthwiseSeparableConv(in_channels, out_channels,
                                              kernel_size, stride, padding,
                                               )
        self.ds_conv1 = DepthwiseSeparableConv(in_channels, out_channels,
                                              kernel_size, stride, padding,
                                              )

        self.ca = ChannelAttention(out_channels, reduction_ratio)
        self.ca1 = ChannelAttention(out_channels, reduction_ratio)

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x,y):
        x = self.ds_conv(x)
        x = self.bn(x)
        x = self.act(x)

        y = self.ds_conv1(y)
        y = self.bn(y)
        y = self.act(y)

        # 应用通道注意力
        attention = self.ca(x)+self.ca1(y)
        x = x * torch.sigmoid(attention)+x
        y = y * torch.sigmoid(attention)+y

        return x,y


############SwinTransformer

from timm.layers import DropPath, to_2tuple, trunc_normal_


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # Get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            self.register_buffer("attn_mask", attn_mask)
        else:
            self.attn_mask = None

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W, C)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W,C):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)
    return x

class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C
        x = self.norm(x)
        x = self.reduction(x)
        return x

class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

class SwinTransformerFeatureInteraction(nn.Module):
    def __init__(self, input_size=(128, 216), in_chans=128, embed_dim=128,
                 depths=[2, 2], num_heads=[4, 8], window_size=4,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()

        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=input_size, patch_size=1, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(len(depths)):
            layer = BasicLayer(dim=int(embed_dim),
                               input_resolution=(patches_resolution[0],
                                                 patches_resolution[1]),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)

        # Feature interaction module
        self.interaction = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        return x

    def forward(self, x1, x2):
        """Forward pass for feature interaction

        Args:
            x1: First input feature of shape (B, C, H, W)
            x2: Second input feature of shape (B, C, H, W)

        Returns:
            Output feature of same shape as inputs after interaction
        """
        # Process both features through SwinTransformer
        B,C,H,W=x1.shape
        x11 = self.forward_features(x1)  # B L C
        x22 = self.forward_features(x2)  # B L C

        # Feature interaction
        x = torch.cat([x11, x22], dim=-1)  # B L 2C
        x = self.interaction(x)  # B L C

        # Reshape back to original spatial dimensions
        x = x.reshape(B, C, H, W)
        x1=x1+x
        x2=x2+x

        return x1,x2


####Deformable DETR

from torch.nn.init import xavier_uniform_, constant_


class DeformableFeatureInteraction(nn.Module):
    def __init__(self, channels=216, num_heads=8, num_points=4):
        """
        Args:
            channels: 输入特征图的通道数 (默认216)
            num_heads: 注意力头的数量 (默认8)
            num_points: 每个注意力头的采样点数 (默认4)
        """
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.num_points = num_points
        self.head_dim = channels // num_heads

        # 确保通道数能被注意力头整除
        assert self.head_dim * num_heads == channels, "channels必须能被num_heads整除"

        # 用于生成采样偏移和注意力权重的卷积层
        self.offset_conv = nn.Conv2d(channels, num_heads * num_points * 3, kernel_size=3, padding=1)

        # 输出投影层
        self.output_proj = nn.Conv2d(channels, channels, kernel_size=1)

        # 初始化权重
        self._reset_parameters()

    def _reset_parameters(self):
        # 初始化偏移卷积层
        xavier_uniform_(self.offset_conv.weight)
        constant_(self.offset_conv.bias, 0.)
        # 初始化输出投影层
        xavier_uniform_(self.output_proj.weight)
        constant_(self.output_proj.bias, 0.)

    def forward(self, feat1, feat2):
        """
        Args:
            feat1: 第一个输入特征图 [batch, H, W, C]
            feat2: 第二个输入特征图 [batch, H, W, C]

        Returns:
            交互后的特征图 [batch, H, W, C]
        """
        # 确保输入形状相同
        assert feat1.shape == feat2.shape, "输入特征图形状必须相同"
        batch, H, W, C = feat1.shape

        # 转换为PyTorch标准格式 [batch, C, H, W]
        feat1 = feat1.permute(0, 3, 1, 2).contiguous()
        feat2 = feat2.permute(0, 3, 1, 2).contiguous()

        # 生成参考点网格 (归一化到[0,1])
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5 / H, 1 - 0.5 / H, H, device=feat1.device),
            torch.linspace(0.5 / W, 1 - 0.5 / W, W, device=feat1.device)
        )
        ref_points = torch.stack((ref_x, ref_y), dim=-1)  # [H, W, 2]
        ref_points = ref_points.unsqueeze(0).repeat(batch, 1, 1, 1)  # [batch, H, W, 2]

        # 通过卷积生成偏移量和注意力权重
        offset_params = self.offset_conv(feat1)  # [batch, num_heads*num_points*3, H, W]
        offset_params = offset_params.permute(0, 2, 3, 1)  # [batch, H, W, num_heads*num_points*3]

        # 分解为偏移量和注意力权重
        offsets = offset_params[..., :self.num_heads * self.num_points * 2]  # 2D偏移量
        attn_weights = offset_params[..., self.num_heads * self.num_points * 2:]  # 注意力权重

        # 重塑为合适的形状
        offsets = offsets.view(batch, H, W, self.num_heads, self.num_points, 2)
        attn_weights = attn_weights.view(batch, H, W, self.num_heads, self.num_points)
        attn_weights = F.softmax(attn_weights, dim=-1)  # 在采样点上归一化

        # 计算采样位置 = 参考点 + 偏移量
        sample_points = ref_points.unsqueeze(3).unsqueeze(4) + offsets
        sample_points = sample_points.clamp(0, 1)  # 限制在[0,1]范围内

        # 在feat2上采样特征
        sampled_feat = self._bilinear_sample(feat2, sample_points)  # [batch, H, W, num_heads, num_points, head_dim]

        # 应用注意力权重并聚合
        weighted_feat = sampled_feat * attn_weights.unsqueeze(-1)  # [batch, H, W, num_heads, num_points, head_dim]
        aggregated_feat = weighted_feat.sum(dim=-2)  # [batch, H, W, num_heads, head_dim]

        # 合并多头输出
        aggregated_feat = aggregated_feat.view(batch, H, W, -1)  # [batch, H, W, C]

        # 残差连接 + 输出投影
        output = feat1.permute(0, 2, 3, 1) + aggregated_feat[:,:,:,:216]  # [batch, H, W, C]
        output = output.permute(0, 3, 1, 2)  # [batch, C, H, W]
        output = self.output_proj(output)
        feat1=feat1+output
        feat2=feat2+output

        # 恢复原始格式 [batch, H, W, C]
        return feat1.permute(0, 2, 3, 1),feat2.permute(0, 2, 3, 1)

    def _bilinear_sample(self, feat, sample_points):
        """
        双线性采样特征图

        Args:
            feat: 特征图 [batch, C, H, W]
            sample_points: 采样点 [batch, H, W, num_heads, num_points, 2]

        Returns:
            采样后的特征 [batch, H, W, num_heads, num_points, head_dim]
        """
        batch, _, feat_H, feat_W = feat.shape
        _, H, W, num_heads, num_points, _ = sample_points.shape

        # 将采样点转换为网格坐标 (-1到1范围)
        grid = sample_points * 2 - 1  # 从[0,1]映射到[-1,1]

        # 重塑特征图以匹配多头结构
        feat = feat.view(batch, self.num_heads, self.head_dim, feat_H, feat_W)

        # 重塑网格以匹配F.grid_sample的输入格式
        grid = grid.reshape(batch, H * W * num_heads * num_points, 1, 2)

        # 采样特征
        sampled = F.grid_sample(
            feat.reshape(batch, self.num_heads * self.head_dim, feat_H, feat_W),
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )  # [batch, num_heads*head_dim, H*W*num_heads*num_points, 1]

        # 重塑为原始维度

        sampled = sampled.view(batch, self.num_heads, self.head_dim*4, H, W, self.num_heads)

        # 调整维度顺序 [batch, H, W, num_heads, num_points, head_dim]
        sampled = sampled.permute(0, 3, 4, 1, 5, 2).contiguous()
        return sampled.reshape(batch, H, W, num_heads, num_points, -1)




class Fusion3(nn.Module):
    def __init__(self, pseudo_in, valid_in, outplanes):
        super(Fusion3, self).__init__()

        # self.attention0000 =DSConvWithChannelAttention(pseudo_in,pseudo_in)
        # self.attention0000 =SwinTransformerFeatureInteraction()
        # self.attention0000 = DeformableFeatureInteraction(channels=216,num_heads=8)
        # self.attention0001 = DeformableFeatureInteraction(channels=216,num_heads=8)
        # self.attention0002 = DeformableFeatureInteraction(channels=216,num_heads=8)
        # self.attention0003 = DeformableFeatureInteraction(channels=216,num_heads=8)


        self.attention0 = TransAttention(channels=pseudo_in)
        self.attention1 = ROIAttention(channels=pseudo_in)
        self.attention2 = ROIAttention(channels=pseudo_in)

        self.conv1 = torch.nn.Conv1d(valid_in * 2, outplanes, 1)  #128+128,256,1
        self.bn1 = torch.nn.BatchNorm1d(outplanes)
        self.relu = nn.ReLU()

    def forward(self, valid_features, pseudo_features):

        # pseudo_features, valid_features = self.attention0000(pseudo_features.unsqueeze(0), valid_features.unsqueeze(0))
        # pseudo_features, valid_features = self.attention0001(pseudo_features, valid_features)
        # pseudo_features, valid_features = self.attention0002(pseudo_features, valid_features)
        # pseudo_features, valid_features = self.attention0003(pseudo_features, valid_features)

        # pseudo_features=pseudo_features.squeeze(0)
        # valid_features = valid_features.squeeze(0)

        pseudo_features, valid_features = self.attention0(pseudo_features, valid_features)
        pseudo_features, valid_features = self.attention1(pseudo_features, valid_features)
        pseudo_features, valid_features = self.attention2(pseudo_features, valid_features)

        # fusion_features = torch.cat([valid_features2, valid_features, pseudo_features2, pseudo_features], dim=1)
        fusion_features = torch.cat([valid_features, pseudo_features], dim=1)
        fusion_features = self.relu(self.bn1(self.conv1(fusion_features)))

        return fusion_features


