import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
# from pcdet.models.backbones_3d.pointnet2_backbone import PointNet2Backbone, PointNet2MSG
import torch.nn.functional as F
from torch.nn.functional import normalize
from mamba_ssm import Block_2 as MambaBlock
from .codiff_part_diffv1 import CondiDifFusion
from .codiff_vtrans import Dino2Vtrans
import open3d as o3d
from einops import rearrange

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

    def create_open3d_point_cloud(self, point_cloud,channel_feature):
        pcd = o3d.geometry.PointCloud()
        # 设置点的坐标
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        # 设置点的颜色 (RGB)
        pcd.colors = o3d.utility.Vector3dVector(channel_feature)  #
        return pcd

    def save_festures(self, features, frame_idss, position_fea):
        path_f = '/home/gaopan/sfd-mpcd/View_Features/'
        path_fe = '/home/gaopan/sfd-mpcd/View_Features/feature/'
        # torch.save(features, path_f + 'features_tensor.pt')
        tensor = frame_idss[-1]
        idsval = str(tensor)
        # print(idsval)

        import numpy as np
        channel_feature = features.detach().cpu().numpy()


        position_fea=position_fea.detach().cpu().numpy()

        # first_batch_features = torch.mean(first_batch_features, dim=1)
        # channel_feature = first_batch_features.detach().cpu().numpy()
        ##normalize
        min_val=np.min(channel_feature)
        max_val=np.max(channel_feature)

        channel_feature=(channel_feature)/(max_val+1e-8)

        cmap = plt.get_cmap('jet')
        channel_feature = cmap(channel_feature)[:, :3]
        # print(channel_feature.shape)
        # print(channel_feature[0:100,:])

        # channel_feature=1-channel_feature

        pcd = self.create_open3d_point_cloud(position_fea,channel_feature)
        # 显示点云
        o3d.visualization.draw_geometries([pcd])


    def forward(self, color_point_fea, color_point_link,frame_idss,points_coords_src):
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

        ####Show visible map
        # fea2_1_show = torch.zeros_like(fea2_1[:, 0]).to(fea2_1.device)
        # if 52 < float(frame_idss[-1]) < 54:
        #     for nid in range(0,5,1): #4
        #         fea2_show = color_point_fea[:,  0]
        #         fea2_1_show = fea2_1[:, nid]/(fea2_1[:, nid].max()) + fea2_1_show
        #         fea2_1_show[fea2_1_show>1.3]=0
        #
        #         print("id_1：", nid)
        #         # print("fea2_show",fea2_show.max())
        #         print("fea2_1_show", fea2_1_show.max())
        #         # print("fea2_show:", fea2_show.shape)
        #         # print("color_point_fea[:, :3]:", color_point_fea[:, :3].shape)
        #
        #     self.save_festures(fea2_show[1::6], frame_idss, points_coords_src[1::6,:])
        #     print("id_2：", nid)
        #     self.save_festures(fea2_1_show[1::6], frame_idss, points_coords_src[1::6,:])


        return color_conv_fea


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

        keep_mask_y = torch.zeros(216*Bb, dtype=torch.bool, device=x.device)  # 创建全为 False 的布尔数组
        keep_mask_y[0::4] = True  # 每四行中的第1行
        keep_mask_y[3::4] = True  # 每四行中的第4行


        # 使用布尔掩码选择要保留的行
        x_row = x[keep_mask, :]
        y_col = y[:, keep_mask_y[:216]]

        keep_mask[:] = False
        keep_mask[1::4] = True  # 每四行中的第2行
        keep_mask[2::4] = True  # 每四行中的第3行

        keep_mask_y[:] = False
        keep_mask_y[1::4] = True  # 每四行中的第2行
        keep_mask_y[2::4] = True  # 每四行中的第3行

        x_row_trans = x[keep_mask, :]
        x_row_trans = x_row_trans.flip(0)  # 行反转

        y_col_trans = y[:, keep_mask_y[:216]]
        y_col_trans = y_col_trans.flip(1)  # 列反转

        maps= {"x_row": torch.flatten(x_row),
               "x_row_trans":torch.flatten(x_row_trans),
               "y_col": torch.flatten(y_col),
               "y_col_trans":torch.flatten(y_col_trans)
               }

        return maps

    def forward(self, x_fusion_0):

        Hh, Cl, Wl = x_fusion_0.shape #128*256*216
        # x_fusion_0 =torch.cat((x_lid,x_img),dim=1) #128*256*216
        # x_fusion_0=self.conv_1(x_fusion_0)


        x_fusion = x_fusion_0.permute(0,2,1).reshape(-1, Cl) #(128*216)*256
        x_fusion = self.mlp_silu1(x_fusion) #(B*128*216)*256

        with torch.no_grad():
            indexes_x= torch.arange(Hh*216)
            indexes_x = indexes_x.reshape(Hh, 216) # 将一维张量重塑为 (128, 216) 的矩阵
            maps =self.fea_xy_index(indexes_x, 1,Hh)

        keys = [("x_row", "x_row_trans"), ("y_col", "y_col_trans")]

        for i, (key_row, key_row_trans) in enumerate(keys):
            x_features = torch.cat((x_fusion[maps[key_row]], x_fusion[maps[key_row_trans]]), dim=0)
            x_features = x_features.reshape(-1, 1*216, Cl) #B*(128*216)*256
            x_features = self.block[i](x_features) +  x_features
            x_features = x_features.reshape(-1, Cl)  #(128*216)*(256)

            x_fusion[maps[key_row], :] = x_features[:(Hh*108), :]
            x_fusion[maps[key_row_trans], :] = x_features[(Hh*108):, :] #(128*216)*(256)

        x_fusion = self.mlp_silu2(x_fusion).reshape(Hh, -1, 1).permute(0, 2, 1)#128*1*216

        # x_fusion_0=self.mlp_silu2(x_fusion_0.permute(0, 2, 1)).permute(0, 2, 1)
        x_fusion = torch.sigmoid(x_fusion)*x_fusion_0 + x_fusion_0

        return x_fusion



class Fusion3(nn.Module):

    def save_festures(self, features, frame_idss):
        path_f = '/home/gaopan/sfd-mpcd/View_Features/'
        path_fe = '/home/gaopan/sfd-mpcd/View_Features/feature/'
        # torch.save(features, path_f + 'features_tensor.pt')
        tensor = frame_idss[-1]
        idsval = str(tensor)
        # print(idsval)

        first_batch_features = features
        import numpy as np
        if 52 < float(idsval) < 54:

            # for channel_idx in range(first_batch_features.shape[0]):
            for channel_idx in range(1):
                channel_feature = first_batch_features[:, 5, :].detach().cpu().numpy()

                # first_batch_features = torch.mean(first_batch_features, dim=1)
                # channel_feature = first_batch_features.detach().cpu().numpy()
                ##normalize
                min_val=np.min(channel_feature)
                max_val=np.max(channel_feature)
                channel_feature=(channel_feature-min_val)/(max_val-min_val)
                print(channel_feature)

                ###show
                heatmaps=plt.imshow(channel_feature, cmap='jet') #jet hot cool plasma
                plt.xticks(fontsize=22,fontfamily='Arial')
                plt.yticks(fontsize=22,fontfamily='Arial')
                cbar=plt.colorbar(heatmaps) #,shrink=0.8
                cbar.set_label('Value Range')
                # plt.title(idsval+f'_'+f'Channel {channel_idx + 1} Feature Map')
                # plt.savefig(path_fe +f'_'+ idsval+ f'ROI_channel_{channel_idx}_feature_map.png')
                plt.show()

    def save_distribution(self, features, frame_idss):
            # path_f = '/home/gaopan/sfd-mpcd/View_distribution/'
            # path_fe = '/home/gaopan/sfd-mpcd/View_distribution/feature/'
            # torch.save(features, path_f + 'features_tensor.pt')
            tensor = frame_idss[-1]
            idsval = str(tensor)
            # print(idsval)

            first_batch_features = features
            import numpy as np
            if 52 < float(idsval) < 54:

                # for channel_idx in range(first_batch_features.shape[0]):
                for channel_idx in range(1):
                    channel_feature = first_batch_features[:, :, :].detach().cpu().numpy()

                    # first_batch_features = torch.mean(first_batch_features, dim=1)
                    # channel_feature = first_batch_features.detach().cpu().numpy()
                    ##normalize
                    min_val = np.min(channel_feature)
                    max_val = np.max(channel_feature)
                    channel_feature_0 = (channel_feature - min_val) / (max_val - min_val)

                    channel_feature=channel_feature_0[channel_feature_0>0.1]

                    mean_value = channel_feature.mean()
                    std_value = channel_feature.std()

                    print("Mean:", mean_value.item())
                    print("Standard Deviation:", std_value.item())

                    #
                    values = channel_feature.flatten()
                    print(values)
                    plt.hist(values, bins=10, range=(values.min(), values.max()), edgecolor='black')
                    plt.title("Normalized Feature Map Value Distribution")
                    plt.xlabel("Value")
                    plt.ylabel("Frequency")
                    plt.show()
                    ###show
                    # heatmaps = plt.imshow(channel_feature, cmap='jet')  # jet hot cool plasma
                    # cbar = plt.colorbar(heatmaps)  # ,shrink=0.8
                    # cbar.set_label('Value Range')
                    # plt.title(idsval + f'_' + f'Channel {channel_idx + 1} Feature Map')
                    # plt.savefig(path_fe + f'_' + idsval + f'ROI_channel_{channel_idx}_feature_map.png')
                    # plt.show()

    def __init__(self, pseudo_in, valid_in, outplanes, Mamba_CFG, CoDiff_CFG,Train_Mode):  # 128,128,256
        super(Fusion3, self).__init__()

        self.attention1 = ROIAttention(channels=pseudo_in)

        self.attention2 = ROIAttention(channels=pseudo_in)

        if Train_Mode == 'difftrain' or Train_Mode == 'finetune':
            print('bbbbbbbbbbbbbbbbbbbbbbbbb')
            self.attentionb = CondiDifFusion(CoDiff_CFG)

            ####The second train method for 'difftrain'
            # pretrained_diff_path = CoDiff_CFG['pretrained_diff_path']
            #
            # # ----- 加载预训练权重 -------------------------------------------------------------
            # if pretrained_diff_path is not None:
            #     if not os.path.exists(pretrained_diff_path):
            #         raise FileNotFoundError(f"Pretrained diff checkpoint not found: {pretrained_diff_path}")
            #     print(f"Loading diffusion model weights from {pretrained_diff_path}")
            #     checkpoint = torch.load(pretrained_diff_path, map_location='cpu')
            #     # 注意：单独训练时保存的 'model_state_dict' 是不带 "module." 前缀的原始 CondiDifFusion 参数
            #     self.attentionb.load_state_dict(checkpoint['model_state_dict'])
            #     print("Successfully loaded diffusion pretrained weights.")

        if Train_Mode == 'finetune':
            print('aaaaaaaaaaaaaaaaaaaaaaaaaaa')
            self.attentionc = Dino2Vtrans(patch_size=(6, 8), embed_dim=48)
            self.attentiond = MambaFu(Mamba_CFG, in_channel=outplanes)

        self.conv1 = torch.nn.Conv1d(valid_in * 2, outplanes, 1)  # 128+128,256,1
        self.bn1 = torch.nn.BatchNorm1d(outplanes)
        self.relu = nn.ReLU()



    def forward(self, valid_features, pseudo_features, frame_idss, train_mode):
        C, H, W = valid_features.shape
        if self.training:
            B = C // 128
        else:
            B = C // 28 #32

        pseudo_features, valid_features = self.attention1(pseudo_features, valid_features)
        pseudo_features, valid_features = self.attention2(pseudo_features, valid_features)

        if self.training & (train_mode in ['difftrain', 'finetune']):
            pseudo_features = rearrange(pseudo_features, '(b c) h w -> b h c w', b=B)
            valid_features = rearrange(valid_features, '(b c) h w -> b h c w', b=B)

            min_val, max_val = pseudo_features.min(dim=1, keepdim=True)[0], pseudo_features.max(dim=1, keepdim=True)[0]
            pseudo_features_norm = 2 * (pseudo_features - min_val) / (max_val - min_val + 1e-8) - 1

            min_val1, max_val1 = valid_features.min(dim=1, keepdim=True)[0], valid_features.max(dim=1, keepdim=True)[0]
            valid_features_norm = 2 * (valid_features - min_val1) / (max_val1 - min_val1 + 1e-8) - 1

            ##========== 保存特征tensor用于后续扩散模型训练 ==========
            # import os
            # # 创建保存目录
            # save_dir = "/home/gaopan/GP_Third_Method/MPCF_Codiff/data/nuS_for_coDIFF"
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
            ### =====================================================

            loss_diff = self.attentionb(valid_features_norm, pseudo_features_norm) + \
                        self.attentionb(pseudo_features_norm, valid_features_norm)

            pseudo_features_norm = rearrange(((pseudo_features_norm + 1) / 2) * max_val, 'b h c w -> (b c) h w')
            valid_features_norm = rearrange(((valid_features_norm + 1) / 2) * max_val1, 'b h c w -> (b c) h w')

        elif train_mode == 'train':
            loss_diff, pseudo_features_norm, valid_features_norm = 0, pseudo_features, valid_features
        else:
            pseudo_features = rearrange(pseudo_features, '(b c) h w -> b h c w', b=B)
            valid_features = rearrange(valid_features, '(b c) h w -> b h c w', b=B)

            max_val, max_val1 = pseudo_features.max(dim=1, keepdim=True)[0], valid_features.max(dim=1, keepdim=True)[0]
            pseudo_features_norm = 2 * pseudo_features / (max_val + 1e-8) - 1
            valid_features_norm = 2 * valid_features / (max_val1 + 1e-8) - 1

            loss_diff = 0

            ### Denoise_sample
            out, outp = self.attentionb.denoise_sample(valid_features_norm, pseudo_features_norm), \
                self.attentionb.denoise_sample(pseudo_features_norm, valid_features_norm)
            #######For val set
            if self.training:
                pseudo_features_norm = rearrange(
                    ((pseudo_features_norm + 1) / 2) * max_val * 1.1 - outp * (max_val1 * 0.28),
                    'b h c w -> (b c) h w')

            else:
                pseudo_features_norm = rearrange(
                ((pseudo_features_norm + 1) / 2) * max_val * 1.7 - outp * (max_val1 * 0.28 + max_val * 0.6),
                'b h c w -> (b c) h w')

            valid_features_norm = rearrange(out * max_val1 * 1.6 - ((valid_features_norm + 1) / 2) * max_val1 * 0.36,
                                            'b h c w -> (b c) h w')
            #########For test set when inferencing
            # pseudo_features_norm = rearrange(((pseudo_features_norm + 1) / 2) * max_val*(-0.2)  + outp * max_val*1.2,
            #                                 'b h c w -> (b c) h w')
            #
            # valid_features_norm = rearrange(  ((valid_features_norm + 1) / 2) * max_val1*(-0.2) + out * max_val1*1.2,
            #                                 'b h c w -> (b c) h w')
            #
        if train_mode == 'finetune':
            fusion_features_0 = torch.cat([valid_features_norm, pseudo_features_norm], dim=1)
            fusion_features_0_copy1 = fusion_features_0.clone()

            #b:B;h1=32;h2=8;h3=6;w=36

            fusion_features_0 = rearrange(fusion_features_0, '(b c) (h1 h2) (h3 w) -> (b c h1) w (h2 h3)',b=B, h1=32, h3=6)
            fusion_features_0_copy2 = fusion_features_0.clone()

            fusion_tokens_list = self.attentionc(fusion_features_0)

            alfa = torch.sigmoid(fusion_tokens_list[1][:, :, 1:, :])
            fusion_features_0 = alfa * fusion_tokens_list[0][:, :, 1:, :] + (1.0 - alfa) * fusion_features_0_copy2.reshape(-1,32,36,48)

            fusion_features_0 = rearrange(fusion_features_0, '(b c) h1 w (h2 h3) -> (b c) (h1 h2) (h3 w)', b=B, h1=32, h3=6)

            fusion_features_0_copy3 = fusion_features_0.clone()
            fusion_features_1=self.attentiond(fusion_features_0)
            alfa = torch.sigmoid(fusion_features_1)
            fusion_features_0 = alfa * fusion_features_0_copy3 + (1.0 - alfa) * fusion_features_0_copy1

            fusion_features_0 = self.relu(self.bn1(self.conv1(fusion_features_0)))

        else:
            fusion_features_0 = torch.cat([valid_features_norm, pseudo_features_norm], dim=1)
            fusion_features_0 = self.relu(self.bn1(self.conv1(fusion_features_0)))

        return loss_diff, fusion_features_0


