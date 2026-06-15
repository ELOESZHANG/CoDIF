import os
import torch
import torch.nn as nn
from mamba_ssm import Block2 as MambaBlock
from pcdet.models.backbones_2d.fuser.codiff_part_diffv1 import CondiDifFusion
from pcdet.models.backbones_2d.fuser.codiff_vtrans import Dino2Vtrans
from einops import rearrange

#CoDiffusion
class CoDiffusion(nn.Module):
    def __init__(self, model_cfg) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        in_channel = self.model_cfg.IN_CHANNEL #80
        out_channel = self.model_cfg.OUT_CHANNEL #128
        self.conv = nn.Sequential(
            nn.Conv2d(80 + 128, out_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True)
        )


        Train_Mode=self.model_cfg.TRAIN_MODE
        CoDiff_CFG=self.model_cfg.CoDiff_CFG
        Mamba_CFG =self.model_cfg.Mamba_CFG
        patch_size=self.model_cfg.Dino_CFG['PATCH_SIZE']
        embed_dim =self.model_cfg.Dino_CFG['EMBED_DIM']

        self.preconv1 = nn.Sequential(
            nn.Linear(80, out_channel // 2),
        )
        self.preconv2 = nn.Sequential(
            nn.Linear(out_channel, out_channel // 2),
        )
        self.compre1 = nn.Linear(360, 180)
        self.expan1 = nn.Linear(180, 360)

        self.conv1 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True)
        )

        if Train_Mode == 'difftrain' or Train_Mode == 'finetune':

            self.attentionb = CondiDifFusion(CoDiff_CFG)
            print('aaaaaaaaaaaaaaaaaaaaaaaaaaa')
            pretrained_diff_path= CoDiff_CFG['pretrained_diff_path']

            # ----- 加载预训练权重 --------------------------------------------
            if pretrained_diff_path is not None:
                if not os.path.exists(pretrained_diff_path):
                    raise FileNotFoundError(f"Pretrained diff checkpoint not found: {pretrained_diff_path}")
                print(f"Loading diffusion model weights from {pretrained_diff_path}")
                checkpoint = torch.load(pretrained_diff_path, map_location='cpu')
                # 注意：单独训练时保存的 'model_state_dict' 是不带 "module." 前缀的原始 CondiDifFusion 参数
                self.attentionb.load_state_dict(checkpoint['model_state_dict'])
                print("Successfully loaded diffusion pretrained weights.")
            # -------------------------------------------------------------------

        if Train_Mode == 'finetune':
            print('bbbbbbbbbbbbbbbbbbbbbbbbbbbb')
            self.attentionc = Dino2Vtrans(patch_size=patch_size, embed_dim=embed_dim)
            self.attentiond = MambaFu(Mamba_CFG, in_channel=out_channel)

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                spatial_features_img (tensor): Bev features from image modality
                spatial_features (tensor): Bev features from lidar modality

        Returns:
            batch_dict:
                spatial_features (tensor): Bev features after muli-modal fusion
        """
        train_mode=self.model_cfg.TRAIN_MODE
        img_bev = batch_dict['spatial_features_img'] #80
        lidar_bev = batch_dict['spatial_features']  #128
        fusion_features_0 = torch.cat([img_bev, lidar_bev], dim=1)
        fusion_features_0 = self.conv(fusion_features_0)
        fusion_features_0_copy1 = fusion_features_0.clone()


        # print('img_bev', img_bev.shape)   #2,64,180,180
        # print('lidar_bev', lidar_bev.shape)

        if self.training & (train_mode in ['difftrain', 'finetune']):
            # 逐像素置信度门控：特征标准差越大 → 信息量越大 → 该模态更可靠
            img_std = img_bev.std(dim=1, keepdim=True)      # (B,1,H,W)
            lidar_std = lidar_bev.std(dim=1, keepdim=True)  # (B,1,H,W)
            conf_gate = torch.sigmoid(img_std - lidar_std)   # >0.5: 图像更可靠, <0.5: 雷达更可靠
            
            img_bev = self.compre1(img_bev)
            img_bev = self.compre1(rearrange(img_bev, 'b c h w -> b c w h'))
            lidar_bev = self.compre1(lidar_bev)
            lidar_bev = self.compre1(rearrange(lidar_bev, 'b c h w -> b c w h'))
            fusion_features_0_copy1 = self.compre1(fusion_features_0_copy1)
            fusion_features_0_copy1 = self.compre1(rearrange(fusion_features_0_copy1, 'b c h w -> b c w h'))

            img_bev = rearrange(self.preconv1(img_bev.permute(0, 3, 2, 1)), 'b h w c -> b c h w')
            lidar_bev = rearrange(self.preconv2(lidar_bev.permute(0, 3, 2, 1)), 'b h w c -> b c h w')

            max_val =  img_bev.max(dim=1, keepdim=True)[0]
            pseudo_features_norm = 2 * (img_bev ) / (max_val + 1e-8) - 1

            max_val1 = lidar_bev.max(dim=1, keepdim=True)[0]
            valid_features_norm = 2 * (lidar_bev) / (max_val1  + 1e-8) - 1

            # ========== 保存特征tensor用于后续扩散模型训练 ==========
            # import os
            # # 创建保存目录
            # save_dir = "/home/gaopan226/GP_Third_method/MambaFusion/data/nuS_for_Diffuison"
            # os.makedirs(save_dir, exist_ok=True)
            #
            # # 使用一个全局变量
            # if not hasattr(self, 'save_counter'):
            #     self.save_counter = 0
            # # 格式化保存文件名
            # save_idx = str(self.save_counter).zfill(6)
            # if 1 <= self.save_counter <= 70000:  # 000001-020100范围
            #     # 保存pseudo特征（整个batch）
            #     pseudo_save_path = os.path.join(save_dir, f"pseudo_{save_idx}.pt")
            #     torch.save(pseudo_features_norm.detach().cpu(), pseudo_save_path)
            #
            #     # 保存valid特征（整个batch）
            #     valid_save_path = os.path.join(save_dir, f"valid_{save_idx}.pt")
            #     torch.save(valid_features_norm.detach().cpu(), valid_save_path)
            #
            #     print(f"Saved features to {pseudo_save_path} and {valid_save_path}")
            # # 更新计数器
            # self.save_counter += 1
            # # =====================================================

            loss_diff = self.attentionb(valid_features_norm, pseudo_features_norm) + \
                        self.attentionb(pseudo_features_norm, valid_features_norm)

            pseudo_features_norm = ((pseudo_features_norm + 1) / 2) * max_val
            valid_features_norm =((valid_features_norm + 1) / 2) * max_val1
            fusion_features_1 = torch.cat([pseudo_features_norm, valid_features_norm], dim=1)
            fusion_features_1 = self.conv1(fusion_features_1)

        elif train_mode == 'train':
            img_bev = self.compre1(img_bev)
            img_bev = self.compre1(rearrange(img_bev, 'b c h w -> b c w h'))
            lidar_bev = self.compre1(lidar_bev)
            lidar_bev = self.compre1(rearrange(lidar_bev, 'b c h w -> b c w h'))
            fusion_features_0_copy1 = self.compre1(fusion_features_0_copy1)
            fusion_features_0_copy1 = self.compre1(rearrange(fusion_features_0_copy1, 'b c h w -> b c w h'))

            img_bev = rearrange(self.preconv1(img_bev.permute(0, 3, 2, 1)), 'b h w c -> b c h w')
            lidar_bev = rearrange(self.preconv2(lidar_bev.permute(0, 3, 2, 1)), 'b h w c -> b c h w')

            max_val = img_bev.max(dim=1, keepdim=True)[0]
            pseudo_features_norm = 2 * (img_bev) / (max_val + 1e-8) - 1

            max_val1 = lidar_bev.max(dim=1, keepdim=True)[0]
            valid_features_norm = 2 * (lidar_bev) / (max_val1 + 1e-8) - 1

            loss_diff = 0

            pseudo_features_norm = ((pseudo_features_norm + 1) / 2) * max_val
            valid_features_norm = ((valid_features_norm + 1) / 2) * max_val1
            fusion_features_1 = torch.cat([pseudo_features_norm, valid_features_norm], dim=1)
            fusion_features_1 = self.conv1(fusion_features_1)

        else:
            # 逐像素置信度门控：特征标准差越大 → 信息量越大 → 该模态更可靠
            img_std = img_bev.std(dim=1, keepdim=True)      # (B,1,H,W)
            lidar_std = lidar_bev.std(dim=1, keepdim=True)  # (B,1,H,W)
            conf_gate = torch.sigmoid(img_std - lidar_std)   # >0.5: 图像更可靠, <0.5: 雷达更可靠

            img_bev = self.compre1(img_bev)
            img_bev = self.compre1(rearrange(img_bev, 'b c h w -> b c w h'))
            lidar_bev = self.compre1(lidar_bev)
            lidar_bev = self.compre1(rearrange(lidar_bev, 'b c h w -> b c w h'))

            fusion_features_0_copy1 = self.compre1(fusion_features_0_copy1)
            fusion_features_0_copy1 = self.compre1(rearrange(fusion_features_0_copy1, 'b c h w -> b c w h'))

            img_bev = rearrange(self.preconv1(img_bev.permute(0, 3, 2, 1)), 'b h w c -> b c h w')
            lidar_bev = rearrange(self.preconv2(lidar_bev.permute(0, 3, 2, 1)), 'b h w c -> b c h w')

            max_val, max_val1 = img_bev.max(dim=1, keepdim=True)[0], lidar_bev.max(dim=1, keepdim=True)[0]
            pseudo_features_norm = 2 * img_bev / (max_val + 1e-8) - 1
            valid_features_norm = 2 * lidar_bev / (max_val1 + 1e-8) - 1

            loss_diff = 0          
            ### V+P:75.32-73.08  ;V+V+0.5:75.34-73.08 ; V+V+0.8:75.34-73.10 ;V+V+1.0:75.31-73.05; V+V+0.7:75.32-73.06;
            out, outp = 0.8*(self.attentionb.denoise_sample(valid_features_norm, pseudo_features_norm) +valid_features_norm ), \
                        0.8*(self.attentionb.denoise_sample(pseudo_features_norm, valid_features_norm) +pseudo_features_norm)


            pseudo_features_norm = ((outp + 1) / 2) * max_val
            valid_features_norm =((out + 1) / 2) * max_val1
            fusion_features_1 = torch.cat([pseudo_features_norm, valid_features_norm], dim=1)
            fusion_features_1 = self.conv1(fusion_features_1)


        if train_mode == 'finetune':

            fusion_features_0_copy2 = self.attentionc(fusion_features_1)
            alfa = torch.sigmoid(fusion_features_0_copy2)
            fusion_features_1 = alfa * fusion_features_0_copy1.permute(0,1,3,2)+ (1.0 - alfa) * fusion_features_1

            fusion_features_1 = self.attentiond(rearrange(fusion_features_1,'b c h w -> (b h) w c'))
            fusion_features_1 = rearrange(fusion_features_1,'(b h) w c -> b c h w', h=180)
            fusion_features_1 = self.expan1(fusion_features_1)
            fusion_features_1 = rearrange(self.expan1(fusion_features_1.permute(0, 1, 3, 2)), 'b c w h -> b c h w')
            fusion_features_1 =  conf_gate * fusion_features_1 + (1-conf_gate)* fusion_features_0
            # print('fusion_features_1', fusion_features_1.shape)  # [2, 128, 180, 180])

        elif train_mode == 'train':
            fusion_features_1 = self.expan1(fusion_features_1)
            fusion_features_1 = rearrange(self.expan1(fusion_features_1.permute(0, 1, 3, 2)), 'b c w h -> b c h w')+  fusion_features_0
        else:
            fusion_features_1 = self.expan1(fusion_features_1)
            fusion_features_1 = rearrange(self.expan1(fusion_features_1.permute(0, 1, 3, 2)), 'b c w h -> b c h w')
            # 自适应门控融合（替代固定系数）
            # 对噪声/标定误差/振动工况鲁棒：不可靠模态的特征被自动压制：
            # w conf_gate: 75.31-73.05; w/o conf_gate:75.33-73.05     SOTA：75.34-73.09
            fusion_features_1 =  fusion_features_1 + fusion_features_0


         # [2, 128, 360, 360]
        batch_dict['spatial_features'] = fusion_features_1
        batch_dict['loss_diff']=loss_diff
        batch_dict['train_mode']=train_mode
        return batch_dict

class MambaFu(nn.Module):
    def __init__(self,Mamba_CFG,in_channel) -> None:
        super().__init__()
        out_channel = in_channel
        operator_cfg=Mamba_CFG

        self.mlp_silu1 = nn.Sequential(
            nn.Linear(in_channel, out_channel),
            # nn.Linear(out_channel, out_channel),
            nn.SiLU(),
        )

        self.block = nn.ModuleList([
            MambaBlock(**{**operator_cfg, 'layer_id': i+1, 'n_layer': 2, 'with_cp': False, 'd_model': out_channel})
            for i in range(2)
        ])

        self.mlp_silu2 = nn.Sequential(
            nn.LayerNorm(out_channel),
            nn.Linear(out_channel, 1),
            # nn.SiLU(),
            # nn.Dropout(0.1),
            # nn.Linear(out_channel, out_channel),
        )
        # self.para1=nn.Parameter(torch.rand(1))

    def fea_xy_index(self, x, Bb,H):

        # 获取偶数行（从左到右）和奇数行（从右到左）
        y= x.clone()
        x[3::4, :] = x[3::4, :].flip(dims=[1])
        x[1::4, :] = x[1::4, :].flip(dims=[1])

        # 对 y 进行列反转
        y[:, 3::4] = y[:, 3::4].flip(dims=[0])
        y[:, 1::4] = y[:, 1::4].flip(dims=[0])

        # 创建一个布尔掩码，标记要保留的行
        keep_mask = torch.zeros(H*Bb, dtype=torch.bool, device=x.device)  # 创建全为 False 的布尔数组
        keep_mask[0::4] = True  # 每四行中的第1行
        keep_mask[3::4] = True  # 每四行中的第4行

        keep_mask_y = torch.zeros(180*Bb, dtype=torch.bool, device=x.device)  # 创建全为 False 的布尔数组
        keep_mask_y[0::4] = True  # 每四行中的第1行
        keep_mask_y[3::4] = True  # 每四行中的第4行


        # 使用布尔掩码选择要保留的行
        x_row = x[keep_mask, :]
        y_col = y[:, keep_mask_y[:180]]

        keep_mask[:] = False
        keep_mask[1::4] = True  # 每四行中的第2行
        keep_mask[2::4] = True  # 每四行中的第3行

        keep_mask_y[:] = False
        keep_mask_y[1::4] = True  # 每四行中的第2行
        keep_mask_y[2::4] = True  # 每四行中的第3行

        x_row_trans = x[keep_mask, :]
        x_row_trans = x_row_trans.flip(0)  # 行反转

        y_col_trans = y[:, keep_mask_y[:180]]
        y_col_trans = y_col_trans.flip(1)  # 列反转

        maps= {"x_row": torch.flatten(x_row),
               "x_row_trans":torch.flatten(x_row_trans),
               "y_col": torch.flatten(y_col),
               "y_col_trans":torch.flatten(y_col_trans)
               }

        return maps

    def forward(self, x_fusion_0):

        Hh, Wl,Cl  = x_fusion_0.shape #360,180,128
        # x_fusion_0 =torch.cat((x_lid,x_img),dim=1) #128*256*216
        # x_fusion_0=self.conv_1(x_fusion_0)


        x_fusion = x_fusion_0.reshape(-1, Cl) #(256*180)*180
        x_fusion = self.mlp_silu1(x_fusion) #(B*180*180)*256

        with torch.no_grad():
            indexes_x= torch.arange(Hh*180)
            indexes_x = indexes_x.reshape(Hh, 180) # 将一维张量重塑为 (128, 216) 的矩阵
            maps =self.fea_xy_index(indexes_x, 1,Hh)

        keys = [("x_row", "x_row_trans"), ("y_col", "y_col_trans")]

        for i, (key_row, key_row_trans) in enumerate(keys):
            x_features = torch.cat((x_fusion[maps[key_row]], x_fusion[maps[key_row_trans]]), dim=0)
            x_features = x_features.reshape(-1, 1*180, Cl) #B*(128*216)*256
            x_features = self.block[i](x_features) +  x_features
            x_features = x_features.reshape(-1, Cl)  #(128*216)*(256)

            x_fusion[maps[key_row], :] = x_features[:(Hh*90), :]
            x_fusion[maps[key_row_trans], :] = x_features[(Hh*90):, :] #(128*216)*(256)

        x_fusion = self.mlp_silu2(x_fusion).reshape(Hh, -1, 1)#128*1*216

        x_fusion = torch.sigmoid(x_fusion)*x_fusion_0 + x_fusion_0

        return x_fusion



# 假设的配置类和模块
class MockConfig:
    def __init__(self):
        self.IN_CHANNEL = 80
        self.OUT_CHANNEL = 128
        self.TRAIN_MODE = 'train'
        self.CoDiff_CFG = {
            'train_steps': 400,
            'sample_stepss': 5
        }
        self.Mamba_CFG = {
            'd_state': 16,
            'd_conv': 4,
            'expand': 1,
            'drop_path': 0.2
         }
        self.Dino_CFG={
            'PATCH_SIZE': [22, 8],
            'EMBED_DIM': 176
        }

        self.patch_size = [22, 8]
        self.embed_dim = 176



if __name__ == '__main__':

# 替换导入的模块
#     with patch('mamba_ssm.Block2', MockMambaFu), \
#         patch('codiff_part_diff.CondiDifFusion', MockCondiDifFusion), \
#         patch('codiff_vtrans.Dino2Vtrans', MockDino2Vtrans):

    config = MockConfig()
    # model = CoDiffusion(config)

    # 测试数据
    batch_dict = {

        'spatial_features_img': torch.randn(2, 80, 360, 360),
        'spatial_features': torch.randn(2, 128, 360, 360)
    }

    # 测试训练模式
    config.TRAIN_MODE = 'train'
    model = CoDiffusion(config)
    model.train()
    result = model(batch_dict)
    assert 'spatial_features' in result
    assert 'loss_diff' in result

    print("train测试通过!")

    # 测试 difftrain 模式
    config.TRAIN_MODE = 'difftrain'
    model1 = CoDiffusion(config)
    model1.train()
    result = model1(batch_dict)
    assert 'spatial_features' in result
    assert 'loss_diff' in result

    print("difftrain测试通过!")

    # 测试 finetune 模式
    config.TRAIN_MODE = 'finetune'
    model2 = CoDiffusion(config)
    result = model2(batch_dict)
    assert 'spatial_features' in result
    assert 'loss_diff' in result


    print("finetune测试通过!")





