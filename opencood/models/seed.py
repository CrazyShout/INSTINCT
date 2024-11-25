# -*- coding: utf-8 -*-
# Author: Yunjiang Xu <yjxu95@stu.suda.edu>, OpenPCDet
# License: TDG-Attribution-NonCommercial-NoDistrib


import torch
import torch.nn as nn


from opencood.models.sub_modules.mean_vfe import MeanVFE
from opencood.models.sub_modules.spconv_backbone_voxelnext import VoxelResBackBone8xVoxelNeXt
from opencood.models.sub_modules.height_compression import NaiveHeightCompression
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.comm_modules.seed_head import SEEDHead


class SEED(nn.Module):
    def __init__(self, args):
        super(SEED, self).__init__()

        # VFE 每个体素中的所有有效点取均值
        self.vfe = MeanVFE(args['vfe'], num_point_features=4)

        # 3D backbone
        self.backbone3d = VoxelResBackBone8xVoxelNeXt(model_cfg=args['backbone_3d'],
                                                      input_channels=4, 
                                                      grid_size=args['grid_size'], # [2016, 800, 50]
                                                      voxel_size=args['voxel_size'], # [0.1, 0.1, 0.1]
                                                      point_cloud_range=args['lidar_range']) # [-100.8, -40, -3.5, 100.8, 40, 1.5]
        # map_to_bev
        self.height_compression = NaiveHeightCompression(model_cfg=args['map_to_bev'])


        # 2D backbone
        is_resnet = args['backbone_2d'].get("resnet", False)
        if is_resnet:
            print("===use res 2d backbone===")
            self.backbone2d = ResNetBEVBackbone(args['backbone_2d'], 256) # or you can use ResNetBEVBackbone, which is stronger
        else:
            print("===use base 2d backbone===")
            self.backbone2d = BaseBEVBackbone(args['backbone_2d'], args['map_to_bev']['feature_num']) # or you can use ResNetBEVBackbone, which is stronger

        self.train_flag = args.get("train_flag", True)

        # dense_head
        self.dense_head = SEEDHead(model_cfg=args['dense_head'], input_channels=args['dense_head']['input_features'], num_class=1, class_names=['vehicle'], grid_size=args['grid_size'],
                                   point_cloud_range=args['lidar_range'], predict_boxes_when_training=False, voxel_size=args['voxel_size'], train_flag=self.train_flag)

    def forward(self, data_dict):
        # torch.cuda.reset_max_memory_allocated()
        voxel_features = data_dict['processed_lidar']['voxel_features'] # (M, 5, 4)
        voxel_coords = data_dict['processed_lidar']['voxel_coords'] # (M, 3)
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points'] # (M, )
        batch_size = data_dict['batch_size']
        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'object_bbx_center': data_dict['object_bbx_center'], # (B, max_num, 7)
                      'object_bbx_mask': data_dict['object_bbx_mask'],                      
                      'batch_size': batch_size}
        batch_dict = self.vfe(batch_dict)
        batch_dict = self.backbone3d(batch_dict)
        batch_dict = self.height_compression(batch_dict)
        batch_dict = self.backbone2d(batch_dict)
        batch_dict = self.dense_head(batch_dict)

        # max_memory_allocated = torch.cuda.max_memory_allocated()

        if self.train_flag: # 之所以不用nn中的training来判断，实际上是因为loss和模型一起计算，解耦有点麻烦，且因为detr-based方法需要大量的loss，如果解耦反而要return大量的东西
            loss_trans, tb_dict = batch_dict['loss'], batch_dict['tb_dict']
            tb_dict = {
                'loss_trans': loss_trans.item(), # 转换成标量
                **tb_dict
            }
            loss = loss_trans
            return loss, tb_dict
        else:
            return batch_dict['final_box_dicts'][0] # return 检测结果，Dict

        print(f"显存最大使用量：{max_memory_allocated / (1024 ** 3):.2f} GB")

        
        spatial_features_2d = batch_dict['spatial_features_2d']

        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)

        psm = self.cls_head(spatial_features_2d)
        rm = self.reg_head(spatial_features_2d)

        output_dict = {'cls_preds': psm,
                       'reg_preds': rm}
                       
        if self.use_dir:
            dm = self.dir_head(spatial_features_2d)
            output_dict.update({'dir_preds': dm})

        return output_dict