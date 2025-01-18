from scipy.optimize import linear_sum_assignment

import torch
from torch import nn


def box_cxcyczlwh_to_xyxyxy(x):
    '''
    将(x,y,z,l,w,h) 变为 (min_x, min_y, min_z, max_x, max_y, max_z)
    '''
    x_c, y_c, z_c, l, w, h = x.unbind(-1)
    b = [
        (x_c - 0.5 * l),
        (y_c - 0.5 * w),
        (z_c - 0.5 * h),
        (x_c + 0.5 * l),
        (y_c + 0.5 * w),
        (z_c + 0.5 * h),
    ]

    return torch.stack(b, dim=-1)


def box_vol_wo_angle(boxes):
    vol = (boxes[:, 3] - boxes[:, 0]) * (boxes[:, 4] - boxes[:, 1]) * (boxes[:, 5] - boxes[:, 2])

    return vol


def box_intersect_wo_angle(boxes1, boxes2):
    ltb = torch.max(boxes1[:, None, :3], boxes2[:, :3])  # [N,M,3]
    rbf = torch.min(boxes1[:, None, 3:], boxes2[:, 3:])  # [N,M,3]

    lwh = (rbf - ltb).clamp(min=0)  # [N,M,3]
    inter = lwh[:, :, 0] * lwh[:, :, 1] * lwh[:, :, 2]  # [N,M]

    return inter


def box_iou_wo_angle(boxes1, boxes2):
    vol1 = box_vol_wo_angle(boxes1)
    vol2 = box_vol_wo_angle(boxes2)
    inter = box_intersect_wo_angle(boxes1, boxes2)

    union = vol1[:, None] + vol2 - inter
    iou = inter / union

    return iou, union


def generalized_box3d_iou(boxes1, boxes2):
    boxes1 = torch.nan_to_num(boxes1)
    boxes2 = torch.nan_to_num(boxes2)

    assert (boxes1[:, 3:] >= boxes1[:, :3]).all()
    assert (boxes2[:, 3:] >= boxes2[:, :3]).all()

    iou, union = box_iou_wo_angle(boxes1, boxes2)

    ltb = torch.min(boxes1[:, None, :3], boxes2[:, :3])  # [N,M,3]
    rbf = torch.max(boxes1[:, None, 3:], boxes2[:, 3:])  # [N,M,3]

    whl = (rbf - ltb).clamp(min=0)  # [N,M,3]
    vol = whl[:, :, 0] * whl[:, :, 1] * whl[:, :, 2]

    return iou - (vol - union) / vol # 标准 IoU - (包围盒体积 - 并集体积) / 包围盒体积


class HungarianMatcher3d(nn.Module):
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, cost_rad: float = 1, decode_func=None):
        super().__init__()
        self.cost_class = cost_class                # 1.0
        self.cost_bbox = cost_bbox                  # 4.0
        self.cost_giou = cost_giou                  # 2.0
        self.cost_rad = cost_rad                    # 4.0

        self.decode_fuc = decode_func
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0 or cost_rad != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        if "topk_indexes" in outputs.keys():
            pred_logits = torch.gather(
                outputs["pred_logits"],
                1,
                outputs["topk_indexes"].expand(-1, -1, outputs["pred_logits"].shape[-1]),
            ) # 根据索引抽取出来1000个 (B, 1000, 1)
            pred_boxes = torch.gather(
                outputs["pred_boxes"],
                1,
                outputs["topk_indexes"].expand(-1, -1, outputs["pred_boxes"].shape[-1]),
            ) # (B, 1000, 7)
            if 'pred_ious' in outputs.keys():
                pred_iou = torch.gather(
                    outputs["pred_ious"],
                    1,
                    outputs["topk_indexes"].expand(-1, -1, outputs["pred_ious"].shape[-1]),
                ) # (B, 1000, 1)
        else:
            pred_logits = outputs["pred_logits"] # (B, 1000, 1)
            pred_boxes = outputs["pred_boxes"] # (B, 1000, 7)

        bs, num_queries = pred_logits.shape[:2]
        # We flatten to compute the cost matrices in a batch
        out_prob = pred_logits.sigmoid() # (B, 1000, 1)
        # ([batch_size, num_queries, 6], [batch_size, num_queries, 2])
        out_bbox = pred_boxes[..., :6]
        out_rad = pred_boxes[..., 6:7]

    # Also concat the target labels and boxes
        # [batch_size, num_target_boxes]
        tgt_ids = [v["labels"] for v in targets] # [(n1,), (n2,)]
        # [batch_size, num_target_boxes, 6]
        tgt_bbox = [v["gt_boxes"][..., :6] for v in targets] # [(n1,6), (n2,6)]
        # [batch_size, num_target_boxes, 2]
        tgt_rad = [v["gt_boxes"][..., 6:7] for v in targets] # [(n1,1), (n2,1)]

        alpha = 0.25
        gamma = 2.0

        indices = []

        for i in range(bs):
            with torch.cuda.amp.autocast(enabled=False): # 禁用自动混合精度, 强制单精度计算，适合高精度需求场景
                out_prob_i = out_prob[i].float()    # (1000, 1)
                out_bbox_i = out_bbox[i].float()    # (1000, 6)
                out_rad_i = out_rad[i].float()      # (1000, 1)
                tgt_bbox_i = tgt_bbox[i].float()    # (n, 6)
                tgt_rad_i = tgt_rad[i].float()      # (n, 1)

                # [num_queries, num_target_boxes]
                cost_giou = -generalized_box3d_iou(
                    box_cxcyczlwh_to_xyxyxy(out_bbox[i]),
                    box_cxcyczlwh_to_xyxyxy(tgt_bbox[i]),
                ) # (1000, n) 取负数表示GIoU越大，代价越小
                # 分类代价计算方式类似Focal Loss，不同的是，这是
                neg_cost_class = (1 - alpha) * (out_prob_i ** gamma) * (-(1 - out_prob_i + 1e-8).log()) # (1000, 1) 负样本分类代价 表示得分越高 代价越高
                pos_cost_class = alpha * ((1 - out_prob_i) ** gamma) * (-(out_prob_i + 1e-8).log()) # (1000, 1) 正样本代价，得分越高，代价越低
                cost_class = pos_cost_class[:, tgt_ids[i]] - neg_cost_class[:, tgt_ids[i]] # 结果shape (1000, n_idx)，tgt_ids为batch中每个样本对应的gt label[(n1,), (n2,)], 在第二维度上筛选，即每个gt都要跟所有的query去计算对应的label损失

                # Compute the L1 cost between boxes
                # [num_queries, num_target_boxes]
                cost_bbox = torch.cdist(out_bbox_i, tgt_bbox_i, p=1) # p = 1 求的是Manhattan距离，=2为Eucliean距离， 为♾️则是Chebyshev距离
                cost_rad = torch.cdist(out_rad_i, tgt_rad_i, p=1)

            # Final cost matrix
            C_i = (
                    self.cost_bbox * cost_bbox
                    + self.cost_class * cost_class
                    + self.cost_giou * cost_giou
                    + self.cost_rad * cost_rad
            ) # （1000， n）代价矩阵
            # [num_queries, num_target_boxes]
            C_i = C_i.view(num_queries, -1).cpu()
            indice = linear_sum_assignment(C_i) # 匈牙利匹配算法找到最小成本匹配，返回的是一个元组，两个元素都是数组，分别表示最佳匹配的行/列索引
            indices.append(indice) # 批次结果

            # for debug  TODO 加载训练的权重发现，IOU 发现匹配后仍有为0的情况，这是误检？
            # from opencood.pcdet_utils.iou3d_nms import iou3d_nms_utils
            # import numpy as np
            # pred_boxes_debug = torch.cat([out_bbox_i, out_rad_i], dim=1)
            # gt_boxes_debug = torch.cat([tgt_bbox_i, tgt_rad_i], dim=1)
            # pred_boxes_debug = self.decode_fuc(pred_boxes_debug)
            # gt_boxes_debug = self.decode_fuc(gt_boxes_debug)
            
            # indices_row = torch.from_numpy(np.array(indice[0]))
            # indices_col = torch.from_numpy(np.array(indice[1]))
            # print("indices_row  is ", indices_row)
            # print("indices_col  is ", indices_col)
            # print("pred_boxes_debug shape is ", pred_boxes_debug.shape)
            # print("gt_boxes_debug shape is ", gt_boxes_debug.shape)

            # pred_boxes_debug_matched = pred_boxes_debug[indices_row]
            # gt_boxes_debug_matched = gt_boxes_debug[indices_col]
            # ious = iou3d_nms_utils.boxes_iou3d_gpu(pred_boxes_debug_matched, gt_boxes_debug_matched)
            
            # print("ious shape is ", ious.shape)
            # ious = torch.diag(ious).detach()
            # print("ious  is ", ious)

            # print("out_prob_i shape is ", out_prob_i.shape)
            # print("tgt_bbox_i shape is ", tgt_bbox_i.shape)
            # print("C_i shape is ", C_i.shape)
            # print("indice  is ", indice)
            # xxx

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices] # 索引数组变张量 由于gt数量远小于object query数量

    def extra_repr(self):
        s = "cost_class={cost_class}, cost_bbox={cost_bbox}, cost_giou={cost_giou}, cost_rad={cost_rad}"

        return s.format(**self.__dict__)
