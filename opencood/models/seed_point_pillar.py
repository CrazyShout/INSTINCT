# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn> Runsheng Xu <rxx3386@ucla.edu>, OpenPCDet
# License: TDG-Attribution-NonCommercial-NoDistrib


import torch
import torch.nn as nn


from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv, AttentionDownsampleConv
from opencood.models.comm_modules.seed_head import SEEDHead


class SEEDPointPillar(nn.Module):
    def __init__(self, args):
        super(SEEDPointPillar, self).__init__()

        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        is_resnet = args['base_bev_backbone'].get("resnet", False)
        if is_resnet:
            print("===use resbackbone===")
            self.backbone = ResNetBEVBackbone(args['base_bev_backbone'], 64) # or you can use ResNetBEVBackbone, which is stronger
        else:
            print("===use basebackbone===")
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
        self.train_flag = args.get("train_flag", True)

        # dense_head
        self.dense_head = SEEDHead(model_cfg=args['dense_head'], input_channels=args['dense_head']['input_features'], num_class=1, class_names=['vehicle'], grid_size=args['point_pillar_scatter']['grid_size'],
                                   point_cloud_range=args['lidar_range'], predict_boxes_when_training=False, voxel_size=args['voxel_size'], train_flag=self.train_flag)
        

    def forward(self, data_dict):

        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        batch_size = data_dict['batch_size']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'object_bbx_center': data_dict['object_bbx_center'], # (B, max_num, 7)
                      'object_bbx_mask': data_dict['object_bbx_mask'],                      
                      'batch_size': batch_size}

        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)

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