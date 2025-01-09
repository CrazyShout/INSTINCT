# -*- coding: utf-8 -*-
# Author: Yunjiang Xu <yjx95@stu.suda.edu>, OpenPCDet
# License: TDG-Attribution-NonCommercial-NoDistrib

import torch
import torch.nn as nn

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.comm_modules.ConQueR_head import ConQueRHead
from opencood.pcdet_utils.iou3d_nms import iou3d_nms_utils


class ConQueRPointPillar(nn.Module):
    def __init__(self, args):
        super(ConQueRPointPillar, self).__init__()

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

        self.train_flag = args.get("train_flag", True)
        
        # dense_head
        self.dense_head = ConQueRHead(model_cfg=args['dense_head'], input_channels=args['dense_head']['input_features'], num_class=1, class_names=['vehicle'], grid_size=args['point_pillar_scatter']['grid_size'],
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

        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)
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
    
