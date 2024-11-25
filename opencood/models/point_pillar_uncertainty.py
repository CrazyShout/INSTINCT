# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib


import torch
import torch.nn as nn
import numpy as np

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone 
from opencood.models.sub_modules.base_bev_backbone_resnet_fpn import ResNetBEVBackboneFPN
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv, AttentionDownsampleConv
from opencood.utils.model_utils import weight_init
from scipy.stats import entropy

def calc_dist_uncert(x, x_mean, x_std):
    
    # return torch.clamp((x - x_mean - x_std), min=0, max=1)
    return torch.relu(x - x_mean - x_std)

def calc_dist_score(x, x_mean, x_std):
    
    # return torch.abs(x - min(1, x_mean + x_std))
    return torch.relu(min(1, x_mean + x_std) - x)

def calc_deviation_ratio(test_cls, test_score, tp_cls_mean = 0.8355, tp_cls_std = 0.1638, tp_score_mean = 0.6425, tp_score_std = 0.1794):

    dr_uncert = tp_cls_mean / (calc_dist_uncert(test_cls, tp_cls_mean, tp_cls_std) + tp_cls_mean)
    dr_score = tp_score_mean / (calc_dist_score(test_score, tp_score_mean, tp_score_std) + tp_score_mean)
    dr = dr_uncert * dr_score

    return dr

class PointPillarUncertainty(nn.Module):
    def __init__(self, args):
        super(PointPillarUncertainty, self).__init__()

        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        # self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)

        is_resnet = args['base_bev_backbone'].get("resnet", False) # default true
        if is_resnet:
            print("===use ResNet as backbone!==")
            self.backbone = ResNetBEVBackbone(args['base_bev_backbone'], 64) # or you can use ResNetBEVBackbone, which is stronger
            # self.backbone = ResNetBEVBackboneFPN(args['base_bev_backbone'], 64) # or you can use ResNetBEVBackbone, which is stronger
        else:
            self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64) # or you can use ResNetBEVBackbone, which is stronger

        self.out_channel = sum(args['base_bev_backbone']['num_upsample_filter'])

        self.shrink_flag = False
        if 'shrink_header' in args:
            print("===use downsample conv to reduce memory===")
            self.shrink_flag = True
            if args['shrink_header']['use_atten']:
                self.shrink_conv = AttentionDownsampleConv(args['shrink_header'])
            else:
                self.shrink_conv = DownsampleConv(args['shrink_header'])
            self.out_channel = args['shrink_header']['dim'][-1]

        self.uncertainty_dim = args['uncertainty_dim'] # dim=3 means x, y, yaw, dim=2 means x, y

        self.cls_head = nn.Conv2d(self.out_channel, args['anchor_number'],
                                kernel_size=1)
        self.reg_head = nn.Conv2d(self.out_channel, 7 * args['anchor_number'],
                                kernel_size=1)

        self.unc_head = nn.Conv2d(self.out_channel, self.uncertainty_dim * args['anchor_number'],
                                    kernel_size=1)
        
        self.re_parameterization = args.get('re_parameterization', False)

        if self.re_parameterization is True: 
            print("===re-parameterization trick==")
            self.unc_head_cls = nn.Conv2d(self.out_channel, args['anchor_number'],
                                        kernel_size=1) 
        if 'dir_args' in args.keys():
            self.use_dir = True
            self.dir_head = nn.Conv2d(self.out_channel, args['dir_args']['num_bins'] * args['anchor_number'],
                                  kernel_size=1) # BIN_NUM = 2
        else:
            self.use_dir = False

        self.inference_state = False
        self.inference_num = 0
        self.simple_dropout = False
        if 'mc_dropout' in args.keys():
            print("===use dropout to regulate output! dropout rate: %.2f==="% (args['mc_dropout']['dropout_rate']))
            if self.simple_dropout:
                self.feature_dropout = nn.Dropout2d(args['mc_dropout']['dropout_rate'])
            if 'inference_stage' in args['mc_dropout'].keys():
                if args['mc_dropout']['inference_stage'] is True:
                    self.inference_state = True
                    self.inference_num = args['mc_dropout']['inference_num']
                    
                    self.tp_score_mean = args['mc_dropout']['tp_score_mean']
                    self.tp_score_std = args['mc_dropout']['tp_score_std']
                    self.tp_data_ucls_mean = args['mc_dropout']['tp_data_ucls_mean']
                    self.tp_data_ucls_std = args['mc_dropout']['tp_data_ucls_std']
                    self.tp_model_ucls_mean = args['mc_dropout']['tp_model_ucls_mean']
                    self.tp_model_ucls_std = args['mc_dropout']['tp_model_ucls_std']
                    self.data_ureg_mean = args['mc_dropout']['data_ureg_mean']
                    self.data_ureg_std = args['mc_dropout']['data_ureg_std']
                    self.model_ureg_mean = args['mc_dropout']['model_ureg_mean']
                    self.model_ureg_std = args['mc_dropout']['model_ureg_std']

                    self.dairv2x = args['mc_dropout'].get("dairv2x", False)
                    self.unc_normalize = args['mc_dropout'].get("unc_normalize", False)
                    if self.unc_normalize:
                        print("===Normalize unc score===")
                    if self.dairv2x:
                        print("===dairv2x use differnt anchor l&w===")
                        self.anchor_l = 4.5
                        self.anchor_w = 2
                    else:
                        self.anchor_l = 3.9
                        self.anchor_w = 1.6
                        
                    print("===Use MC Dropout! infer %d times!=="%(self.inference_num))

        self.apply(weight_init)
    

    def forward(self, data_dict):

        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']



        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points}

        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)

        spatial_features_2d = batch_dict['spatial_features_2d']

        # unc_preds = self.unc_head(spatial_features_2d) # s is log(b) or log(sigma^2)  移动到这里是因为发现下采样会对不确定性量化造成很严重的数值不稳
        # # re-parametrization trick, or var of classification logit is very difficulty to learn -- xyj 2024/5/27
        # if self.re_parameterization is True:
        #     unc_cls_log_var = self.unc_head_cls(spatial_features_2d) # (N, 2, H, W)

        # downsample feature to reduce memory
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)  # (B, 256, H', W')

        if self.simple_dropout:
            spatial_features_2d = self.feature_dropout(spatial_features_2d)

        cls_preds = self.cls_head(spatial_features_2d)
        reg_preds = self.reg_head(spatial_features_2d)

        unc_preds = self.unc_head(spatial_features_2d) # s is log(b) or log(sigma^2)
        # re-parametrization trick, or var of classification logit is very difficulty to learn -- xyj 2024/5/27
        if self.re_parameterization is True:
            unc_cls_log_var = self.unc_head_cls(spatial_features_2d) # (N, 2, H, W)

        # re-parametrization trick, or var of classification logit is very difficulty to learn -- xyj 2024/5/27
        if self.re_parameterization is True and self.training: # 训练阶段是需要添加噪声的，但是验证阶段以及推理阶段则不需要
            unc_cls_log_var = torch.exp(unc_cls_log_var) # 得到方差
            unc_cls_log_var = torch.sqrt(unc_cls_log_var) # 得到标准差
            epsilon = torch.randn_like(unc_cls_log_var).to(unc_cls_log_var.device)
            cls_preds = cls_preds + epsilon * unc_cls_log_var 



        output_dict = {'cls_preds': cls_preds,
                       'reg_preds': reg_preds,
                       'unc_preds': unc_preds}
        
        if self.use_dir:
            dir_preds = self.dir_head(spatial_features_2d)
            output_dict.update({'dir_preds': dir_preds})

        if self.inference_state is True: # 这里已经是推理阶段
            # MC Dropout
            B,_,H0,W0 = cls_preds.shape
            cls_preds_ntimes_tensor = torch.zeros_like(cls_preds, dtype=cls_preds.dtype, device=cls_preds.device).unsqueeze(0).repeat(self.inference_num, 1, 1, 1, 1)
            reg_preds_ntimes_tensor = torch.zeros_like(reg_preds, dtype=reg_preds.dtype, device=reg_preds.device).unsqueeze(0).repeat(self.inference_num, 1, 1, 1, 1)
            unc_preds_ntimes_tensor = torch.zeros_like(unc_preds, dtype=unc_preds.dtype, device=unc_preds.device).unsqueeze(0).repeat(self.inference_num, 1, 1, 1, 1)
            cls_preds_ntimes_tensor[0] = cls_preds
            reg_preds_ntimes_tensor[0] = reg_preds
            unc_preds_ntimes_tensor[0] = torch.exp(unc_preds) # 变回方差
            if self.re_parameterization is True:
                cls_noise_ntimes_tensor = torch.zeros_like(unc_cls_log_var, dtype=unc_cls_log_var.dtype, device=unc_cls_log_var.device).unsqueeze(0).repeat(self.inference_num, 1, 1, 1, 1)
                cls_noise_ntimes_tensor[0] = torch.exp(unc_cls_log_var) # 这是已经被处理过的标准差 要变回到方差 需要平方
                cls_noise_ntimes_tensor[0] = torch.sqrt(cls_noise_ntimes_tensor[0]) # 这是已经被处理过的标准差 要变回到方差 需要平方

            debug_flag = False
            
            if self.use_dir:
                dir_preds_ntimes_tensor = torch.zeros_like(dir_preds, dtype=dir_preds.dtype, device=dir_preds.device).unsqueeze(0).repeat(self.inference_num, 1, 1, 1, 1)
                dir_preds_ntimes_tensor[0] = dir_preds

            if debug_flag:
                print(f"cls_preds_ntimes_tensor shape is {cls_preds_ntimes_tensor.shape}") # torch.Size([10, 1, 2, 100, 176])
                print(f"reg_preds_ntimes_tensor shape is {reg_preds_ntimes_tensor.shape}") # torch.Size([10, 1, 14, 100, 176])
                print(f"dir_preds_ntimes_tensor shape is {dir_preds_ntimes_tensor.shape}") # torch.Size([10, 1, 4, 100, 176])
                print(f"unc_preds_ntimes_tensor shape is {unc_preds_ntimes_tensor.shape}") # torch.Size([10, 1, 6, 100, 176])
                
            for i in range(1, self.inference_num):
                batch_dict = self.backbone(batch_dict)

                spatial_features_2d = batch_dict['spatial_features_2d']

                # downsample feature to reduce memory
                if self.shrink_flag:
                    spatial_features_2d = self.shrink_conv(spatial_features_2d)  # (B, 256, H', W')

                cls_preds_ntimes_tensor[i] = self.cls_head(spatial_features_2d)
                reg_preds_ntimes_tensor[i] = self.reg_head(spatial_features_2d)
                
                if self.use_dir:
                    dir_preds_ntimes_tensor[i] = self.dir_head(spatial_features_2d)

                unc_preds_ntimes_tensor[i] = self.unc_head(spatial_features_2d) # s is log(b) or log(sigma^2)
                unc_preds_ntimes_tensor[i] = torch.exp(unc_preds_ntimes_tensor[i]) # 方差作为回归噪声
                if self.re_parameterization is True:
                    cls_noise_ntimes_tensor[i] = self.unc_head_cls(spatial_features_2d)
                    cls_noise_ntimes_tensor[i] = torch.exp(cls_noise_ntimes_tensor[i]) # 方差作为分类噪声

                    cls_noise_ntimes_tensor[i] = torch.sqrt(cls_noise_ntimes_tensor[i]) # 标准差
                    
                    # epsilon = torch.randn_like(cls_noise_ntimes_tensor[i]).to(cls_noise_ntimes_tensor[i].device)
                    # cls_preds_ntimes_tensor[i] = cls_preds_ntimes_tensor[i] + epsilon * cls_noise_ntimes_tensor[i]
 

                del spatial_features_2d


            cls_preds_mean = torch.mean(cls_preds_ntimes_tensor, dim=0)
            reg_preds_mean = torch.mean(reg_preds_ntimes_tensor, dim=0) # (1, 14, H, W)
            unc_preds_mean = torch.mean(unc_preds_ntimes_tensor, dim=0) # (1, 2*3, H, W) 直接建模回归不确定性
            # 数据uncertainty-回归 side
            d_a_square = self.anchor_l**2 + self.anchor_w**2
            unc_preds_mean = unc_preds_mean.permute(0,2,3,1) # (1, H, W, 2*3) 这会改变内存布局，因此后面必须逆操作
            unc_preds_mean = unc_preds_mean.reshape(2*H0*W0, -1) # (2HW, 3)
            assert unc_preds_mean.shape[1] == 3
            unc_preds_mean[:, :2] *= d_a_square
            unc_preds_mean = torch.sqrt(unc_preds_mean)
            unc_preds_mean = unc_preds_mean.sum(dim=-1, keepdim = True) # (2HW, 1)
            # unc_preds_mean = (unc_preds_mean - 1.5137) / 0.3543 # 标准化
            if self.unc_normalize:
                unc_preds_mean = (unc_preds_mean - self.data_ureg_mean) / self.data_ureg_std # 标准化
            unc_preds_mean = unc_preds_mean.repeat(1, 3) # (2HW, 3) repeat的原因是后处理的时候默认其是三维，懒得改了，做个同步
            unc_preds_mean = unc_preds_mean.reshape(1, H0, W0, -1).permute(0, 3, 1, 2)
            # unc_preds_mean = unc_preds_mean.reshape(1, H0, W0, -1).permute(0, 3, 1, 2)
            # 数据uncertainty-分类 side
            if self.re_parameterization is True:
                cls_noise_mean = torch.mean(cls_noise_ntimes_tensor, dim=0) # (1, 2, H, W)

            if self.use_dir:
                dir_preds_mean = torch.mean(dir_preds_ntimes_tensor, dim=0) 
            reg_preds_var = torch.var(reg_preds_ntimes_tensor, dim=0) # (1, 14, H, W)

            if debug_flag:
                print("===============mean value===============")
                # print(f"reg_preds_mean is {reg_preds_mean[0,:,0,0]}")
                print(f"cls_preds_mean shape is {cls_preds_mean.shape}") # torch.Size([1, 2, 100, 176])
                print(f"reg_preds_mean shape is {reg_preds_mean.shape}") # torch.Size([1, 14, 100, 176])
                print(f"unc_preds_mean shape is {unc_preds_mean.shape}") # torch.Size([1, 6, 100, 176])
                print(f"dir_preds_mean shape is {dir_preds_mean.shape}") # torch.Size([1, 4, 100, 176])
                print(f"reg_preds_var shape is {reg_preds_var.shape}") # torch.Size([1, 14, 100, 176])
                # print(f"reg_preds_var  is {reg_preds_var}") # torch.Size([1, 14, 100, 176])
                
            # 模型uncertainty-分类 side  衡量每个anchor
            cls_score = torch.sigmoid(cls_preds_mean) # (1, 2, H, W)

            if self.re_parameterization is True:
                # cls_noise = calc_deviation_ratio(cls_noise_mean, cls_score, tp_cls_mean = 0.0584, tp_cls_std = 0.0263, tp_score_mean = 0.6652, tp_score_std = 0.1811) # 计算分类偏差比
                if self.unc_normalize:
                    cls_noise = calc_deviation_ratio(cls_noise_mean, cls_score, tp_cls_mean = self.tp_data_ucls_mean, tp_cls_std = self.tp_data_ucls_std, tp_score_mean = self.tp_score_mean, tp_score_std = self.tp_score_std) # 计算分类偏差比
                else:
                    cls_noise = cls_noise_mean
            else:
                print("==close re-parameterzation==")
                cls_noise = torch.zeros_like(cls_score)

            # 计算分类分数的对数
            log_cls_score = torch.log(cls_score) # (B, 2, H, W)
            log_1_cls_score = torch.log(1 - cls_score) # (B, 2, H, W)

            # 计算熵 除以 log(2) 以将结果从 nats 转换为 bits
            unc_epi_cls = -(cls_score * log_cls_score + (1 - cls_score) * log_1_cls_score) / torch.log(torch.tensor(2.0, device=cls_preds.device))           
            # 将熵的张量转换为 cls_preds 的设备
            # unc_epi_cls = unc_epi_cls.to(cls_score.device)
            if self.unc_normalize:
                unc_epi_cls = calc_deviation_ratio(unc_epi_cls, cls_score, tp_cls_mean = self.tp_model_ucls_mean, tp_cls_std = self.tp_model_ucls_std, tp_score_mean = self.tp_score_mean, tp_score_std = self.tp_score_std) # 计算分类偏差比
            else:
                unc_epi_cls = unc_epi_cls.to(cls_score.device)
            # unc_epi_cls = calc_deviation_ratio(unc_epi_cls, cls_score) # 计算分类偏差比
            # unc_epi_cls = calc_deviation_ratio(unc_epi_cls, cls_score, tp_cls_mean = 0.8201, tp_cls_std = 0.1749, tp_score_mean = 0.6484, tp_score_std = 0.1872) # 计算分类偏差比
            # unc_epi_cls = calc_deviation_ratio(unc_epi_cls, cls_score, tp_cls_mean = 0.8112, tp_cls_std = 0.1690, tp_score_mean = 0.6652, tp_score_std = 0.1809) # 计算分类偏差比

            # 模型回归不确定性
            # d_a_square = 1.6**2 + 3.9**2 # anchor的长宽平方和
            reg_preds_var = reg_preds_var.permute(0,2,3,1).reshape(-1,7) # （1，14， H， W） --> (1, H, W, 14) --> (2HW, 7) 
            reg_preds_var[:,:2] *= d_a_square
            unc_epi_reg = torch.sqrt(reg_preds_var)
            unc_epi_reg = unc_epi_reg[:,0] + unc_epi_reg[:,1] + unc_epi_reg[:,6] # (2HW, 1)
            # unc_epi_reg = (unc_epi_reg - 0.0369) / 0.0149 # 标准化
            if self.unc_normalize:
                unc_epi_reg = (unc_epi_reg - self.model_ureg_mean) / self.model_ureg_std # 标准化
            unc_epi_reg = unc_epi_reg.reshape(B, H0, W0, -1).permute(0, 3, 1, 2)
            # unc_epi_reg = unc_epi_reg.reshape(B, -1, H0, W0)

            if debug_flag:
                print("===============uncertainty epistemic value===============")
                print(f"unc_epi_cls shape is {unc_epi_cls.shape}") # torch.Size([1, 2, 100, 176])
                print(f"unc_epi_reg shape is {unc_epi_reg.shape}") # torch.Size([1, 2, 100, 176])
                # print(f"unc_epi_cls  is {unc_epi_cls}") # torch.Size([1, 14, 100, 176])
                # print(f"unc_epi_reg  is {unc_epi_reg}") # torch.Size([1, 14, 100, 176])
                                
            output_dict.update({'cls_preds': cls_preds_mean,
                        'reg_preds': reg_preds_mean,
                        'unc_preds': unc_preds_mean,
                        'cls_noise': cls_noise,
                        'dir_preds': dir_preds_mean,
                        'unc_epi_cls': unc_epi_cls,
                        'unc_epi_reg': unc_epi_reg})

        return output_dict