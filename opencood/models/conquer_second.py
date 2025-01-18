# -*- coding: utf-8 -*-
# Author: Yunjiang Xu <yjx95@stu.suda.edu>, OpenPCDet
# License: TDG-Attribution-NonCommercial-NoDistrib

import torch
import torch.nn as nn

from opencood.models.sub_modules.mean_vfe import MeanVFE
from opencood.models.sub_modules.sparse_backbone_3d import VoxelBackBone8x
from opencood.models.sub_modules.height_compression import HeightCompression
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.comm_modules.ConQueR_head import ConQueRHead
from opencood.pcdet_utils.iou3d_nms import iou3d_nms_utils


class ConQueRSecond(nn.Module):
    def __init__(self, args):
        super(ConQueRSecond, self).__init__()

        # mean_vfe
        self.mean_vfe = MeanVFE(args['vfe'], 4)
        # sparse 3d backbone
        self.backbone_3d = VoxelBackBone8x(args['backbone_3d'],
                                           4, args['grid_size']) # grid 是 2160， 800， 50
        # height compression
        self.height_compression = HeightCompression(args['map_to_bev'])

        # base ben backbone
        is_resnet = args['backbone_2d'].get("resnet", False)
        if is_resnet:
            print("===use resbackbone===")
            self.backbone_2d = ResNetBEVBackbone(args['backbone_2d'], 256) # or you can use ResNetBEVBackbone, which is stronger
        else:
            print("===use basebackbone===")
            self.backbone_2d = BaseBEVBackbone(args['backbone_2d'], 256) # or you can use ResNetBEVBackbone, which is stronger

        self.train_flag = args.get("train_flag", True)
        
        # dense_head
        self.dense_head = ConQueRHead(model_cfg=args['dense_head'], input_channels=args['dense_head']['input_features'], num_class=1, class_names=['vehicle'], grid_size=args['grid_size'],
                                   point_cloud_range=args['lidar_range'], predict_boxes_when_training=False, voxel_size=args['voxel_size'], train_flag=self.train_flag)

    def forward(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features']# m 5 4
        voxel_coords = data_dict['processed_lidar']['voxel_coords'] # m 4
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points'] # m 
        batch_size = voxel_coords[:,0].max() + 1 # batch size is padded in the first idx
        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'object_bbx_center': data_dict['object_bbx_center'], # (B, max_num, 7)
                      'object_bbx_mask': data_dict['object_bbx_mask'],                      
                      'batch_size': batch_size}

        batch_dict = self.mean_vfe(batch_dict) # m 5 4 -> m 4
        batch_dict = self.backbone_3d(batch_dict) 
        batch_dict = self.height_compression(batch_dict)# 降维 压缩高度 得到4个样本，每个样本256特征维度 (256, 100, 252)
        batch_dict = self.backbone_2d(batch_dict)# 会形成两个尺度的256 得到512
        batch_dict = self.dense_head(batch_dict)

        if self.train_flag: # 之所以不用nn中的training来判断，实际上是因为loss和模型一起计算，解耦有点麻烦，且因为detr-based方法需要大量的loss，如果解耦反而要return大量的东西
            loss_trans, tb_dict = batch_dict['loss'], batch_dict['tb_dict']
            tb_dict = {
                'loss_trans': loss_trans.item(), # 转换成标量
                **tb_dict
            }
            loss = loss_trans
            return loss, tb_dict
        else:
            # pred_dicts, recall_dicts = self.post_processing(batch_dict)
            # return pred_dicts, recall_dicts
            return batch_dict['final_box_dicts'][0] # return 检测结果，Dict
        
    @staticmethod
    def generate_recall_record(box_preds, recall_dict, batch_index, data_dict=None, thresh_list=None):
        if 'object_bbx_center' not in data_dict:
            return recall_dict

        rois = data_dict['rois'][batch_index] if 'rois' in data_dict else None 
        gt_boxes = data_dict['object_bbx_center'][batch_index] # (max_num, 100)

        if recall_dict.__len__() == 0:
            recall_dict = {'gt': 0}
            for cur_thresh in thresh_list:
                recall_dict['rcnn_%s' % (str(cur_thresh))] = 0

        cur_gt = gt_boxes # (max_num,7)
        k = cur_gt.__len__() - 1
        while k >= 0 and cur_gt[k].sum() == 0: # 一直求有效的gt
            k -= 1
        cur_gt = cur_gt[:k + 1] # (N, 7) 这是真正有效的gt

        if cur_gt.shape[0] > 0:
            if box_preds.shape[0] > 0: # 预测的有 (M, 7)
                iou3d_rcnn = iou3d_nms_utils.boxes_iou3d_gpu(box_preds[:, 0:7], cur_gt[:, 0:7]) # (N, M)
            else:
                iou3d_rcnn = torch.zeros((0, cur_gt.shape[0]))

            for cur_thresh in thresh_list: # [0.3, 0.5, 0.7]
                if iou3d_rcnn.shape[0] == 0:
                    recall_dict['rcnn_%s' % str(cur_thresh)] += 0
                else:
                    rcnn_recalled = (iou3d_rcnn.max(dim=0)[0] > cur_thresh).sum().item() # M个预测里有几个大于阈值的，记下个数
                    recall_dict['rcnn_%s' % str(cur_thresh)] += rcnn_recalled

            recall_dict['gt'] += cur_gt.shape[0]
        else:
            gt_iou = box_preds.new_zeros(box_preds.shape[0])
        return recall_dict

    def post_processing(self, batch_dict):
        batch_size = batch_dict['batch_size']
        final_pred_dict = batch_dict['final_box_dicts']
        recall_dict = {}
        assert batch_size == 1
        for index in range(batch_size):
            pred_boxes = final_pred_dict[index]['pred_boxes'] # (n,7)
            pred_scores = final_pred_dict[index]['pred_scores'] # (n,)
            pred_labels = final_pred_dict[index]['pred_labels'] # (n,)


            recall_dict = self.generate_recall_record(
                box_preds=final_pred_dict[index]['pred_boxes'],
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=[0.3, 0.5, 0.7]
            )

        return final_pred_dict, recall_dict
    
