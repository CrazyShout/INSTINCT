# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

# This eval utils use correct AP calculation with global sorting of all predictions
# Different from original OPV2V paper. 
# See https://github.com/DerrickXuNu/OpenCOOD/issues/104


import os

import numpy as np
import torch

from opencood.utils import common_utils
from opencood.hypes_yaml import yaml_utils
from opencood.utils import box_utils

def voc_ap(rec, prec):
    """
    VOC 2010 Average Precision.
    """
    rec.insert(0, 0.0)
    rec.append(1.0)
    mrec = rec[:]

    prec.insert(0, 0.0)
    prec.append(0.0)
    mpre = prec[:]

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)

    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mrec, mpre

allnum = 0
total_num = 0
max_distance = 0
min_distance = 366
def caluclate_tp_fp(det_boxes, det_score, gt_boxes, result_stat, iou_thresh, cls_noise = None, uncertainty_score = None, unc_epi_cls = None, unc_epi_reg = None):
    """
    Calculate the true positive and false positive numbers of the current
    frames.
    Parameters
    ----------
    det_boxes : torch.Tensor
        The detection bounding box, shape (N, 8, 3) or (N, 4, 2).
    det_score :torch.Tensor
        The confidence score for each preditect bounding box.
    gt_boxes : torch.Tensor
        The groundtruth bounding box.
    result_stat: dict
        A dictionary contains fp, tp and gt number.
    iou_thresh : float
        The iou thresh.
    """
    # fp, tp and gt in the current frame
    fp = []
    tp = []
    gt = gt_boxes.shape[0]
    if det_boxes is not None:
        # convert bounding boxes to numpy array
        det_boxes = common_utils.torch_tensor_to_numpy(det_boxes) # (N, 8, 3)
        det_score = common_utils.torch_tensor_to_numpy(det_score) # (N, )
        gt_boxes = common_utils.torch_tensor_to_numpy(gt_boxes) # (N_gt, 8, 3)
        center_box = box_utils.corner_to_center(det_boxes)[:, :2] # (N, 2)
        distance2detector = center_box * 2.5
        distance2detector = np.sqrt(np.sum(np.square(distance2detector), axis=1)) # (N,)

        distance2detector_list = distance2detector.tolist()

        # sort the prediction bounding box by score
        score_order_descend = np.argsort(-det_score)
        det_score = det_score[score_order_descend] # from high to low
        det_polygon_list = list(common_utils.convert_format(det_boxes)) # 返回一个列表，有N个元素，每一个为Polygon对象，由2d bev bbx构成
        gt_polygon_list = list(common_utils.convert_format(gt_boxes))

        if uncertainty_score is not None:
            # d_a_square = 1.6**2 + 3.9**2
            uncertainty_score = common_utils.torch_tensor_to_numpy(uncertainty_score) # (N, 3)
            # # uncertainty_score = np.exp(uncertainty_score) # 预测的是log方差，所以这里变回方差 TODO 2024年05月28日 在MC Dropout已经变回方差
            # uncertainty_score[:,:2] *= d_a_square # 乘上这个缩放因子，目标是反应出目标的尺度    
            # uncertainty_score = np.sqrt(uncertainty_score) # 开根号反应标准差，拥有更明确的量纲
            # # uncertainty_score = (uncertainty_score - 1.5425) / 0.3588

            uncertainty_x_tp = []
            uncertainty_y_tp = []
            uncertainty_a_tp = []
            uncertainty_x_fp = []
            uncertainty_y_fp = []
            uncertainty_a_fp = []

            cls_noise = common_utils.torch_tensor_to_numpy(cls_noise) # (N, 1) 方差
            uncertainty_ale_cls_tp = []
            uncertainty_ale_cls_fp = []            

            unc_epi_cls = common_utils.torch_tensor_to_numpy(unc_epi_cls) # (N, 1)
            unc_epi_reg = common_utils.torch_tensor_to_numpy(unc_epi_reg) # (N, 1)
            uncertainty_epi_cls_tp = []
            uncertainty_epi_cls_fp = []
            uncertainty_epi_reg_tp = []
            uncertainty_epi_reg_fp = []


            
            score_tp = []
            distance_tp = []
            distance_tp_abnormal = []

        if iou_thresh == 0.7:
            global total_num
            total_num += score_order_descend.shape[0] # 记录总的检测bbx数量
        # match prediction and gt bounding box, in confidence descending order
        for i in range(score_order_descend.shape[0]): # N times
            det_polygon = det_polygon_list[score_order_descend[i]] # 计算当前置信度最高的
            ious = common_utils.compute_iou(det_polygon, gt_polygon_list)
            if len(gt_polygon_list) == 0 or np.max(ious) < iou_thresh:
                fp.append(1)
                tp.append(0)
                if uncertainty_score is not None:
                    uncertainty_x_fp.append(uncertainty_score[score_order_descend[i]][0])
                    uncertainty_y_fp.append(uncertainty_score[score_order_descend[i]][1])
                    uncertainty_a_fp.append(uncertainty_score[score_order_descend[i]][2])
                    uncertainty_epi_cls_fp.append(unc_epi_cls[score_order_descend[i]][0])
                    uncertainty_epi_reg_fp.append(unc_epi_reg[score_order_descend[i]][0])
                    uncertainty_ale_cls_fp.append(cls_noise[score_order_descend[i]][0])
                    
                    # if (iou_thresh == 0.7) & (det_score[i] > 0.6653) & (uncertainty_score[score_order_descend[i]][0] <= -0.2271) & (cls_noise[score_order_descend[i]][0] >= 0.7891):
                    #     print("===有一个很变态的负样本，置信度得分高，不确定性得分小！===")
                    #     print("置信度得分是：", det_score[i]) 
                    #     print("分类不确定性得分是：", uncertainty_score[score_order_descend[i]][0]) 
                    #     print("回归不确定性得分是：", cls_noise[score_order_descend[i]][0])
                    #     print("IoU是: ", np.max(ious))
                    #     global allnum
                    #     allnum += 1
                    #     print("allnum is ", allnum)

                continue

            fp.append(0)
            tp.append(1)
            if uncertainty_score is not None:
                uncertainty_x_tp.append(uncertainty_score[score_order_descend[i]][0])
                uncertainty_y_tp.append(uncertainty_score[score_order_descend[i]][1])
                uncertainty_a_tp.append(uncertainty_score[score_order_descend[i]][2])
                uncertainty_epi_cls_tp.append(unc_epi_cls[score_order_descend[i]][0])
                uncertainty_epi_reg_tp.append(unc_epi_reg[score_order_descend[i]][0])
                uncertainty_ale_cls_tp.append(cls_noise[score_order_descend[i]][0])

                score_tp.append(det_score[i])
                
                # global max_distance, min_distance
                # if distance2detector[i] > max_distance:
                #     max_distance = distance2detector[i]
                # if distance2detector[i] < min_distance:
                #     min_distance = distance2detector[i]
                # print("最大距离是：", max_distance) 
                # print("最小距离是：", min_distance) 

                if (iou_thresh == 0.7) & (uncertainty_score[score_order_descend[i]][0] >= 1.57) & (cls_noise[score_order_descend[i]][0] <= 0.5114):
                    # print("===有一个异常的样本，不确定性得分高！===")
                    # print("置信度得分是：", det_score[i]) 
                    # print("分类不确定性得分是：", uncertainty_score[score_order_descend[i]][0]) 
                    # print("回归不确定性得分是：", cls_noise[score_order_descend[i]][0])
                    # print("IoU是: ", np.max(ious))
                    # print("距离detector的距离是: ", distance2detector[i])
                    # global allnum
                    # allnum += 1
                    # print(f"allnum is {allnum}/{total_num}")
                    distance_tp_abnormal.append(distance2detector[i])
                else:
                    distance_tp.append(distance2detector[i])

            gt_index = np.argmax(ious)
            gt_polygon_list.pop(gt_index)
        result_stat[iou_thresh]['score'] += det_score.tolist()
    result_stat[iou_thresh]['fp'] += fp
    result_stat[iou_thresh]['tp'] += tp
    result_stat[iou_thresh]['gt'] += gt
    if uncertainty_score is not None:
        result_stat[iou_thresh]['uncertainty_x_fp'] += uncertainty_x_fp
        result_stat[iou_thresh]['uncertainty_y_fp'] += uncertainty_y_fp
        result_stat[iou_thresh]['uncertainty_a_fp'] += uncertainty_a_fp
        result_stat[iou_thresh]['uncertainty_x_tp'] += uncertainty_x_tp
        result_stat[iou_thresh]['uncertainty_y_tp'] += uncertainty_y_tp
        result_stat[iou_thresh]['uncertainty_a_tp'] += uncertainty_a_tp

        result_stat[iou_thresh]['uncertainty_ale_cls_fp'] += uncertainty_ale_cls_fp
        result_stat[iou_thresh]['uncertainty_ale_cls_tp'] += uncertainty_ale_cls_tp

        result_stat[iou_thresh]['uncertainty_epi_cls_fp'] += uncertainty_epi_cls_fp
        result_stat[iou_thresh]['uncertainty_epi_cls_tp'] += uncertainty_epi_cls_tp
        result_stat[iou_thresh]['uncertainty_epi_reg_fp'] += uncertainty_epi_reg_fp
        result_stat[iou_thresh]['uncertainty_epi_reg_tp'] += uncertainty_epi_reg_tp

        result_stat[iou_thresh]['score_tp'] += score_tp
        result_stat[iou_thresh]['distance_tp'] += distance_tp
        result_stat[iou_thresh]['distance_tp_ab'] += distance_tp_abnormal
        
        result_stat[iou_thresh]['distance_all'] += distance2detector_list






def calculate_ap(result_stat, iou):
    """
    Calculate the average precision and recall, and save them into a txt.
    Parameters
    ----------
    result_stat : dict
        A dictionary contains fp, tp and gt number.
    iou : float
    """
    iou_5 = result_stat[iou]

    fp = np.array(iou_5['fp'])
    tp = np.array(iou_5['tp'])
    score = np.array(iou_5['score'])
    assert len(fp) == len(tp) and len(tp) == len(score)

    sorted_index = np.argsort(-score)
    fp = fp[sorted_index].tolist()
    tp = tp[sorted_index].tolist()

    gt_total = iou_5['gt']

    cumsum = 0
    for idx, val in enumerate(fp):
        fp[idx] += cumsum
        cumsum += val

    cumsum = 0
    for idx, val in enumerate(tp):
        tp[idx] += cumsum
        cumsum += val

    rec = tp[:]
    for idx, val in enumerate(tp):
        rec[idx] = float(tp[idx]) / gt_total

    prec = tp[:]
    for idx, val in enumerate(tp):
        prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])

    ap, mrec, mprec = voc_ap(rec[:], prec[:])

    return ap, mrec, mprec


def eval_final_results(result_stat, save_path, infer_info=None):
    dump_dict = {}

    ap_30, mrec_30, mpre_30 = calculate_ap(result_stat, 0.30)
    ap_50, mrec_50, mpre_50 = calculate_ap(result_stat, 0.50)
    ap_70, mrec_70, mpre_70 = calculate_ap(result_stat, 0.70)

    dump_dict.update({'ap30': ap_30,
                      'ap_50': ap_50,
                      'ap_70': ap_70,
                      'mpre_50': mpre_50,
                      'mrec_50': mrec_50,
                      'mpre_70': mpre_70,
                      'mrec_70': mrec_70,
                      })
    if infer_info is None:
        yaml_utils.save_yaml(dump_dict, os.path.join(save_path, 'eval.yaml'))
    else:
        yaml_utils.save_yaml(dump_dict, os.path.join(save_path, f'eval_{infer_info}.yaml'))

    print('The Average Precision at IOU 0.3 is %.4f, '
          'The Average Precision at IOU 0.5 is %.4f, '
          'The Average Precision at IOU 0.7 is %.4f' % (ap_30, ap_50, ap_70))

    return ap_30, ap_50, ap_70