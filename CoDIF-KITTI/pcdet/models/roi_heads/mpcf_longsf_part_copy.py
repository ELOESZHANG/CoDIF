import torch
import torch.nn as nn
import torch.nn.functional as F
# from pcdet.models.backbones_3d.pointnet2_backbone import PointNet2Backbone, PointNet2MSG
import matplotlib.pyplot as plt
from torch.nn.functional import normalize
from mamba_ssm import Block_2 as MambaBlock

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



class UrsusInteract(nn.Module):
    def __init__(self,Mamba_CFG, in_channel) -> None:
        super(UrsusInteract, self).__init__()

        out_channel = in_channel

        # self.conv_init_img = nn.Conv2d(in_channel, out_channel, 3, stride=2,padding=1, bias=False)
        # self.conv_init_lid = nn.Conv2d(out_channel, out_channel, 3, stride=2,padding=1, bias=False)

        operator_cfg= Mamba_CFG

        self.block = nn.ModuleList([
            MambaBlock(**{**operator_cfg, 'layer_id': i+1, 'n_layer': 1, 'with_cp': False, 'd_model': out_channel})
            for i in range(1)
        ])

        self.mlp_silu = nn.Sequential(
            nn.LayerNorm(out_channel),
            nn.Linear(out_channel, out_channel),
            nn.Dropout(0.1),
            nn.SiLU(),
            nn.Linear(out_channel, out_channel),
        )
        self.mlp_silu2 = nn.Sequential(
            nn.Linear(out_channel, out_channel//8),
            nn.Linear(out_channel//8, 4),
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
        # x_row = x_row.flip(0)
        y_col = y[:, keep_mask_y[:216]]
        # y_col = y_col.flip(0)

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


        #########消融实验 Back+fore
        maps= {"x_row": torch.flatten(x_row),
               "x_row_trans":torch.flatten(x_row_trans),
               "y_col": torch.flatten(y_col),
               "y_col_trans":torch.flatten(y_col_trans)
               }

        return maps



    def forward(self,x_img_0,x_lid_0):

        #img_roi: torch.Size([128,128,216])
        #lidar_roi: torch.Size([128,128,216])

        # xxxxoa = x_lid.cpu().detach().numpy()

        Hh ,Cl ,Wl =x_lid_0.shape
        x_img_0 = x_img_0.permute(0, 2, 1)
        x_lid_0 = x_lid_0.permute(0, 2, 1)

        x_img= x_img_0.reshape(-1, Cl) #(128*216)*(Cl)
        x_lid =x_lid_0.reshape(-1, Cl)

        # if self.training:
        with torch.no_grad():
            indexes_x= torch.arange(Hh*216) #128*216
            indexes_x = indexes_x.reshape(Hh, 216) # 将一维张量重塑为 (128, 216) 的矩阵
            maps =self.fea_xy_index(indexes_x, 1,Hh)
        # else:
        #     with torch.no_grad():
        #         indexes_x= torch.arange(Bb*27648)
        #         indexes_x = indexes_x.reshape(Bb*128, 216) # 将一维张量重塑为 (128, 216) 的矩阵
        #         maps =self.fea_xy_index(indexes_x, Bb,128)

        # keys = [("x_row", "x_row_trans"), ("x_row_trans", "x_row"),
        #         ("y_col", "y_col_trans"), ("y_col_trans", "y_col")]

        # keys = [("x_row", "x_row_trans"), ("x_row_trans", "x_row")]
        keys = [("x_row", "x_row_trans")]

        for i, (key_row, key_row_trans) in enumerate(keys):
            x_features = torch.cat((x_lid[maps[key_row]], x_img[maps[key_row_trans]]), dim=0)  #(Bb*256*216)*(Cl)
            x_features = x_features.reshape(-1, 1*216, Cl)
            x_features = self.block[i](x_features) #
            x_features = x_features.reshape(-1, Cl)  #

            x_lid[maps[key_row], :] = x_features[:(Hh*108), :]
            x_img[maps[key_row_trans], :] = x_features[(Hh*108):, :] #(Bb*180*180)*(Cl)

        x_lid = torch.sigmoid(x_lid.reshape(Hh, -1, Cl))*x_lid_0
        x_img = torch.sigmoid(x_img.reshape(Hh, -1, Cl))*x_img_0

        x_lid = self.mlp_silu(x_lid).permute(0, 2, 1)
        x_img = self.mlp_silu(x_img)
        x_pseudo=self.mlp_silu2(x_img)


        return x_img.permute(0, 2, 1), x_lid, maps, x_pseudo.permute(0, 2, 1)

class UrsusFusion(nn.Module):
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
            MambaBlock(**{**operator_cfg, 'layer_id': i+1, 'n_layer': 1, 'with_cp': False, 'd_model': out_channel})
            for i in range(1)
        ])

        self.mlp_silu2 = nn.Sequential(
            nn.LayerNorm(out_channel),
            nn.Linear(out_channel, out_channel//8),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(out_channel//8, 8),
        )
        # self.para1=nn.Parameter(torch.rand(1))

    # def fea_xy_index(self, x, Bb,H):
    #
    #     # 获取偶数行（从左到右）和奇数行（从右到左）
    #     y= x.clone()
    #     x[3::4, :] = x[3::4, :].flip(dims=[1])
    #     x[1::4, :] = x[1::4, :].flip(dims=[1])
    #
    #     # 对 y 进行列反转
    #     y[:, 3::4] = y[:, 3::4].flip(dims=[0])
    #     y[:, 1::4] = y[:, 1::4].flip(dims=[0])
    #
    #     # 创建一个布尔掩码，标记要保留的行
    #     keep_mask = torch.zeros(H*Bb, dtype=torch.bool, device=x.device)  # 创建全为 False 的布尔数组
    #     keep_mask[0::4] = True  # 每四行中的第1行
    #     keep_mask[3::4] = True  # 每四行中的第4行
    #
    #     keep_mask_y = torch.zeros(216*Bb, dtype=torch.bool, device=x.device)  # 创建全为 False 的布尔数组
    #     keep_mask_y[0::4] = True  # 每四行中的第1行
    #     keep_mask_y[3::4] = True  # 每四行中的第4行
    #
    #
    #     # 使用布尔掩码选择要保留的行
    #     x_row = x[keep_mask, :]
    #     y_col = y[:, keep_mask_y[:216]]
    #
    #     keep_mask[:] = False
    #     keep_mask[1::4] = True  # 每四行中的第2行
    #     keep_mask[2::4] = True  # 每四行中的第3行
    #
    #     keep_mask_y[:] = False
    #     keep_mask_y[1::4] = True  # 每四行中的第2行
    #     keep_mask_y[2::4] = True  # 每四行中的第3行
    #
    #     x_row_trans = x[keep_mask, :]
    #     x_row_trans = x_row_trans.flip(0)  # 行反转
    #
    #     y_col_trans = y[:, keep_mask_y[:216]]
    #     y_col_trans = y_col_trans.flip(1)  # 列反转
    #
    #     maps= {"x_row": torch.flatten(x_row),
    #            "x_row_trans":torch.flatten(x_row_trans),
    #            "y_col": torch.flatten(y_col),
    #            "y_col_trans":torch.flatten(y_col_trans)
    #            }
    #
    #     return maps

    def forward(self, x_fusion_0, maps):

        Hh, Cl, Wl = x_fusion_0.shape #128*256*216
        # x_fusion_0 =torch.cat((x_lid,x_img),dim=1) #128*256*216
        # x_fusion_0=self.conv_1(x_fusion_0)

        x_fusion_0= x_fusion_0.permute(0,2,1)
        x_fusion = x_fusion_0.reshape(-1, Cl) #(128*216)*256
        x_fusion = self.mlp_silu1(x_fusion) #(B*128*216)*256

        # with torch.no_grad():
        #     indexes_x= torch.arange(Hh*216)
        #     indexes_x = indexes_x.reshape(Hh, 216) # 将一维张量重塑为 (128, 216) 的矩阵
        #     # maps =self.fea_xy_index(indexes_x, 1,Hh)


        # keys = [("x_row", "x_row_trans"), ("x_row_trans", "x_row"),
        #         ("y_col", "y_col_trans"), ("y_col_trans", "y_col")]
        # keys = [("x_row", "x_row_trans"), ("y_col", "y_col_trans")]
        keys = [("x_row", "x_row_trans")]

        for i, (key_row, key_row_trans) in enumerate(keys):
            x_features = torch.cat((x_fusion[maps[key_row]], x_fusion[maps[key_row_trans]]), dim=0)
            x_features = x_features.reshape(-1, 1*216, Cl) #B*(128*216)*256
            x_features = self.block[i](x_features)
            x_features = x_features.reshape(-1, Cl)  #(128*216)*(256)

            x_fusion[maps[key_row], :] = x_features[:(Hh*108), :]
            x_fusion[maps[key_row_trans], :] = x_features[(Hh*108):, :] #(128*216)*(256)

        x_fusion = torch.sigmoid(x_fusion.reshape(Hh, -1, Cl))*x_fusion_0#

        x_fusion = self.mlp_silu2(x_fusion).permute(0, 2, 1)
        # print(x_fusion.shape)
        # x_fusion = normalize(x_fusion,dim=-1)  + x_fusion_0

        return x_fusion



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


    def __init__(self, pseudo_in, valid_in, outplanes, Mamba_CFG):
        super(Fusion3, self).__init__()

        self.attention1 = ROIAttention(channels=pseudo_in)
        self.attention3 = UrsusInteract(Mamba_CFG, valid_in)
        self.attention2 = ROIAttention(channels=pseudo_in)
        self.attention4 = UrsusFusion(Mamba_CFG,valid_in*2)

        self.conv1 = torch.nn.Conv1d(valid_in * 2, outplanes, 1)  #128+128,256,1
        self.bn1 = torch.nn.BatchNorm1d(outplanes)
        self.relu = nn.ReLU()

    def forward(self, valid_features, pseudo_features,frame_idss):
        #128*128*256 H,C,W to 1*C*H*W

        # nid = 2  # 2
        # cid = 10  # 10
        # if 52 < float(frame_idss[-1]) < 54:
        #     pseudo_features_0 = pseudo_features[nid, cid, :].reshape(6, 6, 6)
        #     valid_features_0 = valid_features[nid, cid, :].reshape(6, 6, 6)
        #     # pseudo_features_0=torch.mean(pseudo_features_0, dim=-1)
        #     print("pseudo_features_0:", pseudo_features_0.shape)
        #     self.save_festures(pseudo_features_0, frame_idss)
        #     self.save_festures(valid_features_0, frame_idss)

        pseudo_features, valid_features       = self.attention1(pseudo_features, valid_features)
        # if 52 < float(frame_idss[-1]) < 54:
        #     pseudo_features_0 = pseudo_features[nid, cid, :].reshape(6, 6, 6)
        #     valid_features_0 = valid_features[nid, cid, :].reshape(6, 6, 6)
        #     self.save_festures(pseudo_features_0, frame_idss)
        #     # self.save_festures(valid_features_0, frame_idss)

        pseudo_features, valid_features, maps, pseudo_fea = self.attention3(pseudo_features, valid_features)
        # if 52 < float(frame_idss[-1]) < 54:
        #     pseudo_features_0 = pseudo_features[nid, cid, :].reshape(6, 6, 6)
        #     valid_features_0 = valid_features[nid, cid, :].reshape(6, 6, 6)
        #     self.save_festures(pseudo_features_0, frame_idss)
        #     # self.save_festures(valid_features_0, frame_idss)

        pseudo_features, valid_features       = self.attention2(pseudo_features, valid_features)
        # if 52 < float(frame_idss[-1]) < 54:
        #     pseudo_features_0 = pseudo_features[nid, cid, :].reshape(6, 6, 6)
        #     valid_features_0 = valid_features[nid, cid, :].reshape(6, 6, 6)
        #     self.save_festures(pseudo_features_0, frame_idss)
        #     # self.save_festures(valid_features_0, frame_idss)

        fusion_features_0 = torch.cat([valid_features, pseudo_features], dim=1)
        fusion_features_0 = self.relu(self.bn1(self.conv1(fusion_features_0)))

        # if 52 < float(frame_idss[-1]) < 54:
        #     fusion_features_00 = fusion_features_0[4, 15, :].reshape(6, 6, 6)
        #     self.save_festures(fusion_features_00, frame_idss)


        fusion_features_0 = self.attention4(fusion_features_0, maps)
        # if 52 < float(frame_idss[-1]) < 54:
        #     fusion_features_00 = fusion_features_0[4, 15, :].reshape(6, 6, 6)
        #     self.save_festures(fusion_features_00, frame_idss)



        return fusion_features_0, pseudo_fea

